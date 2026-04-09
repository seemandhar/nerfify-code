from __future__ import annotations

import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Type

import torch
import torch.distributed as dist
from torch.cuda.amp.grad_scaler import GradScaler
from torch.nn import Parameter
from torch.nn.parallel import DistributedDataParallel as DDP

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.datamanagers.base_datamanager import DataManager, DataManagerConfig
from nerfstudio.models.base_model import ModelConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipeline, VanillaPipelineConfig

from nerfify.methods.ar_nerf.ar_nerf_datamanager import ARNeRFDataManagerConfig, _to_device
from nerfify.methods.ar_nerf.ar_nerf_model import ARNeRFModel, ARNeRFModelConfig


@dataclass
class ARNeRFPipelineConfig(VanillaPipelineConfig):
    _target: Type = field(default_factory=lambda: ARNeRFPipeline)
    datamanager: DataManagerConfig = field(default_factory=ARNeRFDataManagerConfig)
    model: ModelConfig = field(default_factory=ARNeRFModelConfig)


class ARNeRFPipeline(VanillaPipeline):
    """AR-NeRF Pipeline.

    Differences from VanillaPipeline:
    1. Does not assume a nerfstudio dataparser / scene_box.
    2. Propagates global training step into the model (for freq mask / blur schedule).
    3. Eval renders in fixed-size chunks to avoid OOM on full images.
    """

    def __init__(
        self,
        config: ARNeRFPipelineConfig,
        device: str,
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        grad_scaler: Optional[GradScaler] = None,
    ):
        # Call nn.Module.__init__ directly to skip VanillaPipeline.__init__
        super(VanillaPipeline, self).__init__()

        self.config    = config
        self.test_mode = test_mode
        self.world_size = world_size

        # DataManager
        self.datamanager: DataManager = config.datamanager.setup(
            device=device,
            test_mode=test_mode,
            world_size=world_size,
            local_rank=local_rank,
        )
        assert self.datamanager.train_dataset is not None, "Missing training dataset."

        # Dummy scene box — AR-NeRF doesn't use scene_box but ModelConfig needs it
        from nerfstudio.data.scene_box import SceneBox
        dummy_aabb = torch.tensor([[-1.5, -1.5, -1.5], [1.5, 1.5, 1.5]])
        dummy_scene_box = SceneBox(aabb=dummy_aabb)

        self._model = config.model.setup(
            scene_box     = dummy_scene_box,
            num_train_data= len(self.datamanager.train_dataset),
            metadata      = {},
            device        = device,
            grad_scaler   = grad_scaler,
        )
        self.model.to(device)

        if world_size > 1:
            self._model = typing.cast(
                ARNeRFModel,
                DDP(self._model, device_ids=[local_rank], find_unused_parameters=True),
            )
            dist.barrier(device_ids=[local_rank])

    # ------------------------------------------------------------------
    # Step propagation
    # ------------------------------------------------------------------

    def _set_model_step(self, step: int):
        m = self.model   # unwrapped from DDP by the base property
        if hasattr(m, "_step"):
            m._step = step

    # ------------------------------------------------------------------
    # Training  (rays already subsampled by DataManager)
    # ------------------------------------------------------------------

    def get_train_loss_dict(self, step: int):
        self._set_model_step(step)

        batch, _ = self.datamanager.next_train(step)
        # batch["rays"] is (n, 6),  batch["rgb"] is (n, 3)

        ray_bundle = _rays_to_bundle(batch)
        model_outputs = self._model(ray_bundle)

        gt_batch = {"image": batch["rgb"]}   # (n, 3)

        metrics_dict = self.model.get_metrics_dict(model_outputs, gt_batch)
        loss_dict    = self.model.get_loss_dict(model_outputs, gt_batch, metrics_dict)
        return model_outputs, loss_dict, metrics_dict

    # ------------------------------------------------------------------
    # Evaluation (chunk full-image rays to avoid OOM)
    # ------------------------------------------------------------------

    def get_eval_loss_dict(self, step: int):
        self._set_model_step(step)
        self.eval()

        batch, _ = self.datamanager.next_eval(step)
        # Subsample a small window for the loss so it stays fast + memory-safe
        rays = batch["rays"]
        rgb  = batch["rgb"]
        n    = min(self.datamanager.config.eval_chunk_size, rays.shape[0])
        idx  = torch.randperm(rays.shape[0], device=rays.device)[:n]
        sub_batch = {**batch, "rays": rays[idx], "rgb": rgb[idx]}

        ray_bundle   = _rays_to_bundle(sub_batch)
        model_outputs = self.model(ray_bundle)
        gt_batch      = {"image": sub_batch["rgb"]}

        metrics_dict = self.model.get_metrics_dict(model_outputs, gt_batch)
        loss_dict    = self.model.get_loss_dict(model_outputs, gt_batch, metrics_dict)
        self.train()
        return model_outputs, loss_dict, metrics_dict

    def get_eval_image_metrics_and_images(self, step: int):
        """Render a full image in chunks, then compute metrics."""
        self._set_model_step(step)
        self.eval()

        batch, _ = self.datamanager.next_eval_image(step)
        H = int(batch["H"][0]) if isinstance(batch["H"], torch.Tensor) else int(batch["H"])
        W = int(batch["W"][0]) if isinstance(batch["W"], torch.Tensor) else int(batch["W"])

        with torch.no_grad():
            rgb_pred, acc_pred = _render_full_image_chunked(
                rays         = batch["rays"],         # (H*W, 6)
                near_far     = batch["near_far"],
                model        = self.model,
                chunk_size   = self.datamanager.config.eval_chunk_size,
                set_step_fn  = self._set_model_step,
                step         = step,
            )

        # Reshape to image
        rgb_pred_img = rgb_pred.view(H, W, 3)
        acc_pred_img = acc_pred.view(H, W, 1)
        gt_img       = batch["rgb"].view(H, W, 3)

        gt_batch = {"image": batch["rgb"]}
        outputs  = {
            "rgb_fine":          rgb_pred,
            "rgb_coarse":        rgb_pred,
            "accumulation_fine": acc_pred,
            "accumulation_coarse": acc_pred,
            "H": H,
            "W": W,
        }
        metrics_dict, images_dict = self.model.get_image_metrics_and_images(outputs, gt_batch)
        metrics_dict["num_rays"]  = H * W

        images_dict = {}
        # Override images with proper H×W tensors
        images_dict["img"] = torch.cat([gt_img, rgb_pred_img, rgb_pred_img], dim=1)

        self.train()
        return metrics_dict, images_dict

    def get_average_eval_image_metrics(
        self,
        step: Optional[int] = None,
        output_path: Optional[Path] = None,
        get_std: bool = False,
    ):
        self.eval()
        metrics_list = []
        n_eval = len(self.datamanager.eval_dataset)

        for _ in range(n_eval):
            m, _ = self.get_eval_image_metrics_and_images(step or 0)
            metrics_list.append(m)

        agg: Dict[str, float] = {}
        for key in metrics_list[0]:
            vals = torch.tensor([m[key] for m in metrics_list], dtype=torch.float32)
            if get_std:
                std, mean = torch.std_mean(vals)
                agg[key], agg[f"{key}_std"] = float(mean), float(std)
            else:
                agg[key] = float(vals.mean())

        self.train()
        return agg

    # ------------------------------------------------------------------
    # Pipeline interface
    # ------------------------------------------------------------------

    def get_training_callbacks(self, attrs):
        return (
            self.datamanager.get_training_callbacks(attrs)
            + self.model.get_training_callbacks(attrs)
        )

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        return {
            **self.datamanager.get_param_groups(),
            **self.model.get_param_groups(),
        }

    def forward(self):
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rays_to_bundle(batch: Dict[str, Any]) -> RayBundle:
    """Convert batch dict with 'rays' (N,6) key into a nerfstudio RayBundle.

    Normalises directions and attaches near/far planes from `near_far`.
    """
    rays     = batch["rays"]      # (N, 6)
    near_far = batch.get("near_far", None)

    if rays.dim() == 3:
        rays = rays.squeeze(0)    # handle leftover batch dim

    origins    = rays[:, :3]
    directions = rays[:, 3:6]
    directions = directions / (directions.norm(dim=-1, keepdim=True) + 1e-8)

    N      = origins.shape[0]
    dev    = origins.device

    if near_far is not None:
        nf = near_far.squeeze() if near_far.dim() > 1 else near_far
        nears = torch.full((N, 1), float(nf[0]), device=dev)
        fars  = torch.full((N, 1), float(nf[1]), device=dev)
    else:
        nears = torch.zeros(N, 1, device=dev)
        fars  = torch.ones(N, 1, device=dev)

    return RayBundle(
        origins    = origins,
        directions = directions,
        pixel_area = torch.ones(N, 1, device=dev),
        nears      = nears,
        fars       = fars,
    )


def _render_full_image_chunked(
    rays: torch.Tensor,
    near_far: torch.Tensor,
    model: ARNeRFModel,
    chunk_size: int,
    set_step_fn,
    step: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Render all rays in `chunk_size` chunks, concatenate results.

    Returns:
        rgb_pred: (N, 3) fine RGB predictions.
        acc_pred: (N, 1) fine accumulation.
    """
    N          = rays.shape[0]
    rgb_chunks = []
    acc_chunks = []

    for start in range(0, N, chunk_size):
        end        = min(start + chunk_size, N)
        chunk_rays = rays[start:end]

        fake_batch = {"rays": chunk_rays, "near_far": near_far}
        bundle     = _rays_to_bundle(fake_batch)

        out = model(bundle)
        rgb_chunks.append(out["rgb_fine"].detach())
        acc_chunks.append(out["accumulation_fine"].detach())

    return torch.cat(rgb_chunks, dim=0), torch.cat(acc_chunks, dim=0)