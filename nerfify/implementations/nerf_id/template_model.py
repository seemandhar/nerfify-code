"""
Template Model — Nerfacto backbone + NeRF-in-detail learned sampler
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Type

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from nerfstudio.cameras.rays import RayBundle, Frustums, RaySamples
from nerfstudio.model_components.renderers import (
    AccumulationRenderer,
    DepthRenderer,
    RGBRenderer,
)
from nerfstudio.model_components.scene_colliders import NearFarCollider
# from nerfstudio.models.nerfacto import NerfactoModel, NerfactoModelConfig
from nerfstudio.models.vanilla_nerf import NeRFModel, VanillaModelConfig
from nerfstudio.models.base_model import Model
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfify.methods.nerf_id.template_field import TemplateNerfField
from nerfstudio.field_components.encodings import NeRFEncoding
from nerfstudio.model_components.ray_samplers import PDFSampler, UniformSampler

from typing import Optional
import torch
from jaxtyping import Float
from torch import Tensor

from nerfstudio.cameras.rays import RayBundle, RaySamples


def normalized_points_to_ray_samples(
    ray_bundle: RayBundle,
    normalized_points,
    spacing_fn: Optional[callable] = None,
    spacing_fn_inv: Optional[callable] = None,
) -> RaySamples:
    assert ray_bundle.nears is not None, "ray_bundle.nears must be provided"
    assert ray_bundle.fars is not None, "ray_bundle.fars must be provided"
    
    if normalized_points.dim() == ray_bundle.origins.dim() - 1:
        bins = torch.cat([
            torch.zeros_like(normalized_points[..., :1]),
            (normalized_points[..., :-1] + normalized_points[..., 1:]) / 2.0,
            torch.ones_like(normalized_points[..., :1])
        ], dim=-1)
    else:
        bins = normalized_points
    
    if spacing_fn is not None and spacing_fn_inv is not None:
        s_near = spacing_fn(ray_bundle.nears)
        s_far = spacing_fn(ray_bundle.fars)
        
        def spacing_to_euclidean_fn(x):
            return spacing_fn_inv(x * s_far + (1 - x) * s_near)
        
        euclidean_bins = spacing_to_euclidean_fn(bins)
    else:
        def spacing_to_euclidean_fn(x):
            return x * ray_bundle.fars + (1 - x) * ray_bundle.nears
        
        euclidean_bins = spacing_to_euclidean_fn(bins)
    
    ray_samples = ray_bundle.get_ray_samples(
        bin_starts=euclidean_bins[..., :-1, None],
        bin_ends=euclidean_bins[..., 1:, None],
        spacing_starts=bins[..., :-1, None],
        spacing_ends=bins[..., 1:, None],
        spacing_to_euclidean_fn=spacing_to_euclidean_fn,
    )
    
    return ray_samples


def ray_samples_to_normalized_points(
    ray_samples: RaySamples,
    use_bin_centers: bool = True,
):
    assert ray_samples.spacing_starts is not None, "ray_samples must have spacing_starts"
    
    if use_bin_centers:
        assert ray_samples.spacing_ends is not None, "ray_samples must have spacing_ends"
        normalized_points = (ray_samples.spacing_starts[..., 0] + 
                           ray_samples.spacing_ends[..., 0]) / 2.0
    else:
        normalized_points = ray_samples.spacing_starts[..., 0]
    
    return normalized_points


@dataclass
class TemplateModelConfig(VanillaModelConfig):
    """Config for TemplateModel (inherits all Nerfacto knobs)."""
    _target: Type = field(default_factory=lambda: TemplateModel)
    num_coarse_samples: int = 64
    num_fine_samples: int = 192
    proposer_type: str = "mlpmix"  # "mlpmix" or "pool"
    stage_mimic_steps: int = 15000  # first half mimic, second half e2e
    # stage_mimic_steps: int = 100  # first half mimic, second half e2e
    mimic_loss_mult: float = 1.0
    coarse_loss_mult: float = 1.0
    importance_loss_mult: float = 1.0
    importance_threshold: float = 0.03
    use_learned_proposer_in_eval: bool = True
    l2_appearance_mult: float = 0.0


class _RayPoolProposer(nn.Module):
    def __init__(self, in_dim: int, nc: int, nf: int, hidden: int = 256, use_mlpmix: bool = True):
        super().__init__()
        self.nc = nc
        self.nf = nf
        self.point_mlp = nn.Sequential(
            nn.Linear(in_dim + 8, hidden),  # 16-dim sine/cos posenc
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
        )
        self.use_mlpmix = use_mlpmix
        if use_mlpmix:
            self.token_mixer = nn.Sequential(
                nn.LayerNorm([hidden, nc]),
                nn.Conv1d(hidden, hidden, kernel_size=1, groups=1),  # channel-mix
            )
        self.ray_fc = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(inplace=True))
        self.prop_head = nn.Linear(hidden, nf)
        self.importance_head = nn.Linear(hidden, nc + nf)
        self._sigmoid = nn.Sigmoid()

    @staticmethod
    def _posenc(x: Tensor, L: int = 4) -> Tensor:
        freqs = [2.0 ** i for i in range(L)]
        outs = []
        for f in freqs:
            outs.append(torch.sin(2 * np.pi * f * x))
            outs.append(torch.cos(2 * np.pi * f * x))
        return torch.cat(outs, dim=-1)

    def forward(self, coarse_feats: Tensor, pos_norm: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        R, Nc, C = coarse_feats.shape
        pos_enc = self._posenc(pos_norm.view(R, Nc, 1))
        h = torch.cat([coarse_feats, pos_enc], dim=-1)  # [R, Nc, C+16]
        h = self.point_mlp(h)  # [R, Nc, H]

        if self.use_mlpmix:
            h_m = h.permute(0, 2, 1)  # [R, H, Nc]
            h_m = self.token_mixer(h_m)  # conv1d
            h = (h + h_m.permute(0, 2, 1)) * 0.5
        ray_repr = h.mean(dim=1)  # [R, H]
        rr = self.ray_fc(ray_repr)
        proposals = self._sigmoid(self.prop_head(rr))  # [R, Nf] in [0,1]
        importance_logits = self.importance_head(rr.detach())  # stop-grad into coarse feats
        return proposals, rr, importance_logits


class TemplateModel(NeRFModel):

    config: TemplateModelConfig

    def populate_modules(self) -> None:
        super().populate_modules()

        position_encoding = NeRFEncoding(
            in_dim=3, num_frequencies=10, min_freq_exp=0.0, max_freq_exp=8.0, include_input=True
        )
        direction_encoding = NeRFEncoding(
            in_dim=3, num_frequencies=4, min_freq_exp=0.0, max_freq_exp=4.0, include_input=True
        )

        self.sampler_pdf = PDFSampler(num_samples=self.config.num_fine_samples, include_original=False)

        self.field_coarse = TemplateNerfField(
            position_encoding=position_encoding,
            direction_encoding=direction_encoding,
        )

        self.field_fine = TemplateNerfField(
            position_encoding=position_encoding,
            direction_encoding=direction_encoding,
        )

        use_mlpmix = self.config.proposer_type.lower() == "mlpmix"
        self.proposer = _RayPoolProposer(in_dim=256, nc=self.config.num_coarse_samples,
                                             nf=self.config.num_fine_samples, use_mlpmix=use_mlpmix)
        self._step = 0

    def get_training_callbacks(self, training_callback_attributes):
        from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackLocation

        def set_step(step: int):
            self._step = step

        return [
            TrainingCallback(
                where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                update_every_num_iters=1,
                func=set_step,
            )
        ]

    def get_param_groups(self) -> Dict[str, List[torch.nn.Parameter]]:
        param_groups = super().get_param_groups()
        param_groups["proposer"] = list(self.proposer.parameters())
        return param_groups

    def get_outputs_(self, ray_bundle: RayBundle):
        if self.field_coarse is None or self.field_fine is None:
            raise ValueError("populate_fields() must be called before get_outputs")

        ray_samples_uniform = self.sampler_uniform(ray_bundle)
        if self.temporal_distortion is not None:
            offsets = None
            if ray_samples_uniform.times is not None:
                offsets = self.temporal_distortion(
                    ray_samples_uniform.frustums.get_positions(), ray_samples_uniform.times
                )
            ray_samples_uniform.frustums.set_offsets(offsets)

        field_outputs_coarse = self.field_coarse.forward(ray_samples_uniform)
        if self.config.use_gradient_scaling:
            field_outputs_coarse = scale_gradients_by_distance_squared(field_outputs_coarse, ray_samples_uniform)
        weights_coarse = ray_samples_uniform.get_weights(field_outputs_coarse[FieldHeadNames.DENSITY])
        rgb_coarse = self.renderer_rgb(
            rgb=field_outputs_coarse[FieldHeadNames.RGB],
            weights=weights_coarse,
        )
        accumulation_coarse = self.renderer_accumulation(weights_coarse)
        depth_coarse = self.renderer_depth(weights_coarse, ray_samples_uniform)

        ray_samples_pdf = self.sampler_pdf(ray_bundle, ray_samples_uniform, weights_coarse)
        if self.temporal_distortion is not None:
            offsets = None
            if ray_samples_pdf.times is not None:
                offsets = self.temporal_distortion(ray_samples_pdf.frustums.get_positions(), ray_samples_pdf.times)
            ray_samples_pdf.frustums.set_offsets(offsets)

        field_outputs_fine = self.field_fine.forward(ray_samples_pdf)
        if self.config.use_gradient_scaling:
            field_outputs_fine = scale_gradients_by_distance_squared(field_outputs_fine, ray_samples_pdf)
        weights_fine = ray_samples_pdf.get_weights(field_outputs_fine[FieldHeadNames.DENSITY])
        rgb_fine = self.renderer_rgb(
            rgb=field_outputs_fine[FieldHeadNames.RGB],
            weights=weights_fine,
        )
        accumulation_fine = self.renderer_accumulation(weights_fine)
        depth_fine = self.renderer_depth(weights_fine, ray_samples_pdf)

        outputs = {
            "rgb_coarse": rgb_coarse,
            "rgb_fine": rgb_fine,
            "accumulation_coarse": accumulation_coarse,
            "accumulation_fine": accumulation_fine,
            "depth_coarse": depth_coarse,
            "depth_fine": depth_fine,
        }
        return outputs

    # ---- Forward pass ----
    def get_outputs(self, ray_bundle: RayBundle):
        # print(self._steps)
        # exit(0)
        # Collide near/far
        # ray_bundle = self.collider(ray_bundle)
        # assert ray_bundle.nears is not None and ray_bundle.fars is not None
        # near = ray_bundle.nears[..., None]  # [R,1,1]
        # far = ray_bundle.fars[..., None]

        Nc = self.config.num_coarse_samples
        Nf = self.config.num_fine_samples

        ray_samples_uniform = self.sampler_uniform(ray_bundle)
        if self.temporal_distortion is not None:
            offsets = None
            if ray_samples_uniform.times is not None:
                offsets = self.temporal_distortion(
                    ray_samples_uniform.frustums.get_positions(), ray_samples_uniform.times
                )
            ray_samples_uniform.frustums.set_offsets(offsets)

        field_outputs_coarse = self.field_coarse.forward(ray_samples_uniform)

        if self.config.use_gradient_scaling:
            field_outputs_coarse = scale_gradients_by_distance_squared(field_outputs_coarse, ray_samples_uniform)
        weights_coarse = ray_samples_uniform.get_weights(field_outputs_coarse[FieldHeadNames.DENSITY])
        rgb_coarse = self.renderer_rgb(
            rgb=field_outputs_coarse[FieldHeadNames.RGB],
            weights=weights_coarse,
        )
        accumulation_coarse = self.renderer_accumulation(weights_coarse)
        depth_coarse = self.renderer_depth(weights_coarse, ray_samples_uniform)
        
        uniform_points_norm = ray_samples_to_normalized_points(ray_samples_uniform)

        is_mimic = self.training and (self._step < self.config.stage_mimic_steps)

        prop_norm, ray_repr, importance_logits = self.proposer(field_outputs_coarse['emb'], uniform_points_norm)  # [R,Nf] in [0,1]
        prop_norm_sorted, _ = torch.sort(prop_norm, dim=1)
        prop_ray_bundle = normalized_points_to_ray_samples(ray_bundle, prop_norm_sorted)

        heur_ray_samples_pdf = self.sampler_pdf(ray_bundle, ray_samples_uniform, weights_coarse)
        if self.temporal_distortion is not None:
            offsets = None
            if heur_ray_samples_pdf.times is not None:
                offsets = self.temporal_distortion(heur_ray_samples_pdf.frustums.get_positions(), heur_ray_samples_pdf.times)
            heur_ray_samples_pdf.frustums.set_offsets(offsets)

        heur_points_norm = ray_samples_to_normalized_points(heur_ray_samples_pdf)
        heur_points_sorted, _ = torch.sort(heur_points_norm, dim=1)

        use_learned = (self.training and not is_mimic) or (not self.training and self.config.use_learned_proposer_in_eval and self._step > self.config.stage_mimic_steps)
        ray_samples_pdf = prop_ray_bundle if use_learned else heur_ray_samples_pdf

        field_outputs_fine = self.field_fine.forward(ray_samples_pdf)
        if self.config.use_gradient_scaling:
            field_outputs_fine = scale_gradients_by_distance_squared(field_outputs_fine, ray_samples_pdf)
        weights_fine = ray_samples_pdf.get_weights(field_outputs_fine[FieldHeadNames.DENSITY])
        rgb_fine = self.renderer_rgb(
            rgb=field_outputs_fine[FieldHeadNames.RGB],
            weights=weights_fine,
        )
        accumulation_fine = self.renderer_accumulation(weights_fine)
        depth_fine = self.renderer_depth(weights_fine, ray_samples_pdf)

        outputs = {
            "rgb_coarse": rgb_coarse,
            "rgb_fine": rgb_fine,
            "accumulation_coarse": accumulation_coarse,
            "accumulation_fine": accumulation_fine,
            "depth_coarse": depth_coarse,
            "depth_fine": depth_fine,
            "weights_fine": weights_fine,
            "importance_logits": importance_logits,
        }
        if self.training:
            outputs["prop_norm_sorted"] = prop_norm_sorted
            outputs["heur_norm_sorted"] = heur_points_sorted
        return outputs

    def get_loss_dict(self, outputs: Dict, batch: Dict, metrics_dict=None) -> Dict:
        loss_dict = super().get_loss_dict(outputs, batch, metrics_dict)
        image = batch["image"].to(self.device)

        is_mimic = self.training and (self._step < self.config.stage_mimic_steps)
        if self.training and is_mimic and self.config.mimic_loss_mult > 0:
            # Greedy matching in 1D reduces to sorting both sets
            pred = outputs["prop_norm_sorted"]
            target = outputs["heur_norm_sorted"]
            mimic = F.mse_loss(pred, target)
            loss_dict["mimic_loss"] = self.config.mimic_loss_mult * mimic

        if self.training and (not is_mimic) and self.config.importance_loss_mult > 0:
            # Importance supervision: label positives where fine weights exceed threshold
            weights_f = outputs["weights_fine"][..., 0]  # [R,M]
            labels = (weights_f > self.config.importance_threshold).float()
            logits = outputs["importance_logits"]  # [R,M] (Nc+Nf)
            # If sizes mismatch due to any off-by-one, interpolate via trimming/padding
            minM = min(labels.shape[1], logits.shape[1])
            labels = labels[:, :minM]
            logits = logits[:, :minM]
            pos = labels.sum() + 1e-6
            neg = (1.0 - labels).sum() + 1e-6
            pos_weight = neg / pos
            bce = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)
            loss_dict["importance_loss"] = self.config.importance_loss_mult * bce

        return loss_dict