"""
Template DataManager
"""

from dataclasses import dataclass, field
from typing import Dict, Literal, Tuple, Type, Union

import torch

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.datamanagers.base_datamanager import (
    VanillaDataManager,
    VanillaDataManagerConfig,
)
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes, TrainingCallbackLocation



@dataclass
class ARNeRFDataManagerConfig(VanillaDataManagerConfig):
    """Template DataManager Config

    Add your custom datamanager config parameters here.
    """

    _target: Type = field(default_factory=lambda: ARNeRFDataManager)
    sample_annealing = False
    start_samples = 1024
    end_samples = 4096
    max_steps = 1000


class ARNeRFDataManager(VanillaDataManager):
    """BioNeRF DataManager

    Args:
        config: the DataManagerConfig used to instantiate class
    """

    config: ARNeRFDataManagerConfig

    def __init__(
        self,
        config: ARNeRFDataManagerConfig,
        device: Union[torch.device, str] = "cpu",
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        **kwargs,  # pylint: disable=unused-argument
    ):
        super().__init__(
            config=config, device=device, test_mode=test_mode, world_size=world_size, local_rank=local_rank, **kwargs
        )
        self.start_samples = config.start_samples
        self.end_samples = config.end_samples
        self.max_steps = config.max_steps

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ):
        callbacks = super().get_training_callbacks(training_callback_attributes)

        if self.config.sample_annealing:
            def step(step):
                # print(f"{self.train_pixel_sampler.num_rays_per_batch}")
                progress = min(step / self.max_steps, 1.0)
                new_samples = int(
                    self.start_samples + progress * (self.end_samples - self.start_samples)
                )
                self.train_pixel_sampler.set_num_rays_per_batch(new_samples)

            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=step,
                )
            )

        return callbacks

    def next_train(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the train dataloader."""
        self.train_count += 1
        image_batch = next(self.iter_train_image_dataloader)
        assert self.train_pixel_sampler is not None
        assert isinstance(image_batch, dict)
        batch = self.train_pixel_sampler.sample(image_batch)
        ray_indices = batch["indices"]
        ray_bundle = self.train_ray_generator(ray_indices)
        return ray_bundle, batch

# """
# AR-NeRF DataManager — per-scene optimisation on DTU.

# Camera loading and ray generation are ported directly from the FreeNeRF / AR-NeRF
# JAX codebase (datasets.py → class DTU).  Key points that differ from our old
# version:

#   1. Camera matrices are loaded via cv2.decomposeProjectionMatrix from
#      Calibration/cal18/pos_NNN.txt (not from Cameras/train/*.txt).
#   2. pixtocam  = inv(camera_mat)  is stored; rays are built as
#          cam_dirs  = ray_dirs @ pixtocam.T
#          directions = cam_dirs @ cam2world[:3,:3].T
#   3. Poses are recentered (recenter_poses) then rescaled by max |t|.
#   4. The canonical pixelnerf train/test split is used:
#          train_idx = [25, 22, 28, 40, 44, 48, 0, 8, 13]
#          exclude   = [3,4,5,6,7,16,17,18,19,20,21,36,37,38,39]
#          test_idx  = remaining indices
#      `input_views` selects the first N of train_idx.
#   5. Training samples `num_rays_per_batch` random rays from a random training
#      image each step (memory-safe).
#   6. Evaluation returns full images, chunked by the pipeline.
# """

# from __future__ import annotations

# from dataclasses import dataclass, field
# from itertools import cycle
# from typing import Dict, List, Literal, Tuple, Type, Union

# import cv2
# import imageio
# import numpy as np
# import os
# import random
# import torch
# from torch.nn import Parameter
# from torch.utils.data import DataLoader, Dataset

# from nerfstudio.data.datamanagers.base_datamanager import (
#     VanillaDataManager,
#     VanillaDataManagerConfig,
# )
# from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes


# # ---------------------------------------------------------------------------
# # Pose utilities  (ported from datasets.py)
# # ---------------------------------------------------------------------------

# def _pad_poses(p: np.ndarray) -> np.ndarray:
#     """[..., 3, 4] → [..., 4, 4] by appending [0,0,0,1]."""
#     bottom = np.broadcast_to([0., 0., 0., 1.], p[..., :1, :4].shape)
#     return np.concatenate([p[..., :3, :4], bottom], axis=-2)


# def _unpad_poses(p: np.ndarray) -> np.ndarray:
#     """[..., 4, 4] → [..., 3, 4]."""
#     return p[..., :3, :4]


# def _normalize(x: np.ndarray) -> np.ndarray:
#     return x / np.linalg.norm(x)


# def _viewmatrix(lookdir, up, position) -> np.ndarray:
#     vec2 = _normalize(lookdir)
#     vec0 = _normalize(np.cross(up, vec2))
#     vec1 = _normalize(np.cross(vec2, vec0))
#     return np.stack([vec0, vec1, vec2, position], axis=1)


# def _poses_avg(poses: np.ndarray) -> np.ndarray:
#     position = poses[:, :3, 3].mean(0)
#     z_axis   = poses[:, :3, 2].mean(0)
#     up       = poses[:, :3, 1].mean(0)
#     return _viewmatrix(z_axis, up, position)


# def _recenter_poses(poses: np.ndarray) -> np.ndarray:
#     """Recenter all poses around the origin."""
#     cam2world = _poses_avg(poses)
#     poses_out = _unpad_poses(np.linalg.inv(_pad_poses(cam2world)) @ _pad_poses(poses))
#     return poses_out


# def _rescale_poses(poses: np.ndarray) -> np.ndarray:
#     """Rescale so that max |translation| = 1."""
#     s = np.max(np.abs(poses[:, :3, 3]))
#     out = poses.copy()
#     out[:, :3, 3] /= s
#     return out


# # ---------------------------------------------------------------------------
# # DTU camera loader  (mirrors datasets.py DTU._load_renderings)
# # ---------------------------------------------------------------------------

# # Canonical PixelNeRF split (same as FreeNeRF / AR-NeRF JAX code)
# _TRAIN_IDX   = [25, 22, 28, 40, 44, 48, 0, 8, 13]
# _EXCLUDE_IDX = [3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 36, 37, 38, 39]
# _TEST_IDX    = [i for i in range(49) if i not in _TRAIN_IDX + _EXCLUDE_IDX]


# def _load_scene(
#     data_root: str,
#     scene: str,
#     input_views: int,
#     factor: int = 1,
# ) -> Dict:
#     """Load all images + cameras for one DTU scene.

#     Args:
#         data_root: Root of the DTU dataset (contains Rectified/, Calibration/).
#         scene:     Scene folder name, e.g. 'scan114'.
#         input_views: How many training views to use (first N of _TRAIN_IDX).
#         factor:    Downsampling factor (1 = full resolution).

#     Returns dict with keys:
#         images_train, rays_train  — list of len(train_ids) items
#         images_eval,  rays_eval   — list of len(test_ids)  items
#         H, W, near, far
#     """
#     scene_dir = os.path.join(data_root, scene)
#     calib_dir = os.path.join(data_root, "Calibration", "cal18")

#     # ---- 1. Load all 49 images + projection matrices ----
#     all_images    = []
#     all_pixtocams = []
#     all_c2w       = []

#     for i in range(1, 50):   # images are 1-indexed in DTU
#         # Image path
#         img_path = os.path.join(scene_dir, f"rect_{i:03d}_3_r5000.png")
#         img = np.array(imageio.imread(img_path), dtype=np.float32) / 255.0
#         if factor > 1:
#             h_new = img.shape[0] // factor
#             w_new = img.shape[1] // factor
#             img = cv2.resize(img, (w_new, h_new), interpolation=cv2.INTER_AREA)
#         all_images.append(img)

#         # Projection matrix
#         proj_path = os.path.join(calib_dir, f"pos_{i:03d}.txt")
#         with open(proj_path, "rb") as f:
#             projection = np.loadtxt(f, dtype=np.float32)

#         # Decompose P = K [R | t]
#         camera_mat, rot_mat, t = cv2.decomposeProjectionMatrix(projection)[:3]
#         camera_mat = camera_mat / camera_mat[2, 2]

#         pose = np.eye(4, dtype=np.float32)
#         pose[:3, :3] = rot_mat.transpose()
#         pose[:3, 3]  = (t[:3] / t[3])[:, 0]
#         pose = pose[:3]   # (3, 4) cam-to-world
#         all_c2w.append(pose)

#         if factor > 1:
#             camera_mat = (
#                 np.diag([1.0 / factor, 1.0 / factor, 1.0]).astype(np.float32)
#                 @ camera_mat
#             )
#         all_pixtocams.append(np.linalg.inv(camera_mat))   # pixtocam

#     all_images    = np.stack(all_images)       # (49, H, W, 3)
#     all_pixtocams = np.stack(all_pixtocams)    # (49, 3, 3)
#     all_c2w       = np.stack(all_c2w)          # (49, 3, 4)

#     H, W = all_images.shape[1:3]

#     # ---- 2. Recenter + rescale poses (same as JAX code) ----
#     all_c2w = _recenter_poses(all_c2w)
#     all_c2w = _rescale_poses(all_c2w)

#     # ---- 3. Split indices ----
#     train_ids = _TRAIN_IDX[:input_views]   # first N of canonical 9-view set
#     test_ids  = _TEST_IDX

#     print(
#         f"[ARNeRF] Scene '{scene}': "
#         f"train ids {train_ids}, "
#         f"{len(test_ids)} eval views."
#     )

#     # ---- 4. Build rays for each split ----
#     def _make_rays(indices):
#         imgs, rays_list = [], []
#         for idx in indices:
#             img   = all_images[idx]        # (H, W, 3)
#             p2c   = all_pixtocams[idx]     # (3, 3)
#             c2w   = all_c2w[idx]           # (3, 4)

#             rays_o, rays_d = _generate_rays_single(H, W, p2c, c2w)
#             # Flatten to (H*W, 3)
#             rays_o = rays_o.reshape(-1, 3)
#             rays_d = rays_d.reshape(-1, 3)
#             rays   = np.concatenate([rays_o, rays_d], axis=-1).astype(np.float32)

#             imgs.append(img.reshape(-1, 3).astype(np.float32))
#             rays_list.append(rays)
#         return imgs, rays_list

#     images_train, rays_train = _make_rays(train_ids)
#     images_eval,  rays_eval  = _make_rays(test_ids)

#     return {
#         "images_train": images_train,   # list of (H*W, 3) arrays
#         "rays_train":   rays_train,     # list of (H*W, 6) arrays
#         "images_eval":  images_eval,
#         "rays_eval":    rays_eval,
#         "train_ids":    train_ids,
#         "test_ids":     test_ids,
#         "H": H, "W": W,
#         # DTU near/far — fixed per the paper (in scene units after rescaling)
#         "near": 0.1,
#         "far":  5.0,
#     }


# def _generate_rays_single(
#     H: int, W: int,
#     pixtocam: np.ndarray,   # (3, 3)
#     cam2world: np.ndarray,  # (3, 4)
# ) -> Tuple[np.ndarray, np.ndarray]:
#     """Generate rays for a single image. Matches JAX DTU._generate_rays exactly.

#     Ray directions: ray_dirs (pixel coords) → cam_dirs (camera space)
#                     → world-space directions via cam2world[:3,:3].
#     Origins: cam2world[:3, 3] broadcast to every pixel.

#     Returns:
#         origins:    (H, W, 3)
#         directions: (H, W, 3)  (NOT unit-normalised, matching JAX code)
#     """
#     # Pixel-centre coordinates  (+0.5 offset, matching JAX code)
#     x, y = np.meshgrid(
#         np.arange(W, dtype=np.float32) + 0.5,
#         np.arange(H, dtype=np.float32) + 0.5,
#         indexing="xy",
#     )
#     # Homogeneous pixel directions in image space
#     ray_dirs = np.stack([x, y, np.ones_like(x)], axis=-1)   # (H, W, 3)

#     # Into camera space:  cam_dirs[i,j] = pixtocam @ ray_dirs[i,j]
#     cam_dirs   = ray_dirs @ pixtocam.T                        # (H, W, 3)

#     # Into world space
#     directions = cam_dirs @ cam2world[:3, :3].T               # (H, W, 3)
#     origins    = np.broadcast_to(cam2world[:3, 3], directions.shape).copy()

#     return origins, directions


# # ---------------------------------------------------------------------------
# # Torch Datasets
# # ---------------------------------------------------------------------------

# class ARNeRFTrainDataset(Dataset):
#     """Returns full rays for one randomly chosen training view per item.
#     The DataManager subsamples num_rays_per_batch rays from it.
#     """

#     def __init__(self, scene_data: Dict):
#         self.images = scene_data["images_train"]   # list of (H*W, 3)
#         self.rays   = scene_data["rays_train"]     # list of (H*W, 6)
#         self.near   = scene_data["near"]
#         self.far    = scene_data["far"]
#         self.n_views = len(self.images)

#     def __len__(self):
#         # Large enough so DataLoader never runs out during training
#         return self.n_views * 10_000

#     def __getitem__(self, _idx: int) -> Dict:
#         # Uniform random view selection — matches JAX single_image batching
#         view_id = random.randint(0, self.n_views - 1)
#         return {
#             "rays":     self.rays[view_id],              # (H*W, 6)
#             "rgb":      self.images[view_id],            # (H*W, 3)
#             "near_far": np.array([self.near, self.far], dtype=np.float32),
#             "view_id":  np.int32(view_id),
#         }


# class ARNeRFEvalDataset(Dataset):
#     """One full image per item (for evaluation)."""

#     def __init__(self, scene_data: Dict):
#         self.images = scene_data["images_eval"]
#         self.rays   = scene_data["rays_eval"]
#         self.near   = scene_data["near"]
#         self.far    = scene_data["far"]
#         self.H      = scene_data["H"]
#         self.W      = scene_data["W"]

#     def __len__(self):
#         return len(self.images)

#     def __getitem__(self, idx: int) -> Dict:
#         return {
#             "rays":     self.rays[idx],
#             "rgb":      self.images[idx],
#             "near_far": np.array([self.near, self.far], dtype=np.float32),
#             "view_id":  np.int32(idx),
#             "H":        np.int32(self.H),
#             "W":        np.int32(self.W),
#         }


# # ---------------------------------------------------------------------------
# # DataManager
# # ---------------------------------------------------------------------------

# @dataclass
# class ARNeRFDataManagerConfig(VanillaDataManagerConfig):
#     """AR-NeRF per-scene DataManager config."""

#     _target: Type = field(default_factory=lambda: ARNeRFDataManager)

#     # Scene
#     data_root:    str   = "data/dtu"
#     """Root of the DTU dataset (contains scan folders + Calibration/)."""
#     scene:        str   = "scan114"
#     """Scene folder name, e.g. 'scan114'."""
#     input_views:  int   = 3
#     """Number of training views (first N of [25,22,28,40,44,48,0,8,13])."""
#     factor:       int   = 1
#     """Image downsampling factor (1 = full resolution)."""

#     # Ray sampling
#     num_rays_per_batch: int = 4096
#     """Rays per training step — primary GPU memory knob."""
#     eval_chunk_size:    int = 4096
#     """Rays per forward pass during evaluation (chunked)."""


# class ARNeRFDataManager(VanillaDataManager):
#     """AR-NeRF per-scene DataManager.

#     * Loads all 49 DTU images at init using the FreeNeRF camera pipeline.
#     * Training: samples `num_rays_per_batch` random rays from a random
#       training view each step.
#     * Evaluation: returns full images (pipeline renders them in chunks).
#     """

#     config: ARNeRFDataManagerConfig

#     def __init__(
#         self,
#         config: ARNeRFDataManagerConfig,
#         device: Union[torch.device, str] = "cpu",
#         test_mode: Literal["test", "val", "inference"] = "val",
#         world_size: int = 1,
#         local_rank: int = 0,
#         **kwargs,
#     ):
#         # Bypass VanillaDataManager.__init__ (requires nerfstudio dataparser)
#         self.config      = config
#         self.device      = device
#         self.world_size  = world_size
#         self.local_rank  = local_rank
#         self.test_mode   = test_mode
#         self.train_count = 0
#         self.eval_count  = 0

#         # Load scene — mirrors DTU._load_renderings + _generate_rays
#         self._scene_data = _load_scene(
#             data_root   = config.data_root,
#             scene       = config.scene,
#             input_views = config.input_views,
#             factor      = config.factor,
#         )

#         self.train_dataset = ARNeRFTrainDataset(self._scene_data)
#         self.eval_dataset  = ARNeRFEvalDataset(self._scene_data)

#         def _worker_init(wid):
#             np.random.seed(wid)
#             random.seed(wid)

#         self._train_loader = DataLoader(
#             self.train_dataset,
#             batch_size     = 1,
#             shuffle        = True,
#             num_workers    = 2,
#             pin_memory     = True,
#             worker_init_fn = _worker_init,
#             collate_fn     = _collate,
#         )
#         self._eval_loader = DataLoader(
#             self.eval_dataset,
#             batch_size  = 1,
#             shuffle     = False,
#             num_workers = 2,
#             pin_memory  = True,
#             collate_fn  = _collate,
#         )

#         self._train_iter = cycle(self._train_loader)
#         self._eval_iter  = cycle(self._eval_loader)

#     # ------------------------------------------------------------------
#     # next_train — subsample num_rays_per_batch rays from one view
#     # ------------------------------------------------------------------

#     def next_train(self, step: int) -> Tuple[Dict, Dict]:
#         """Fetch one random training view, then pick num_rays_per_batch rays.

#         Matches JAX single_image batching:
#             image_index = random int in [0, n_train_views)
#             ray_indices = random ints in [0, H*W)
#         """
#         self.train_count += 1
#         batch = _to_device(next(self._train_iter), self.device)

#         rays = batch["rays"].squeeze(0)    # (H*W, 6)
#         rgb  = batch["rgb"].squeeze(0)     # (H*W, 3)

#         n   = min(self.config.num_rays_per_batch, rays.shape[0])
#         idx = torch.randperm(rays.shape[0], device=self.device)[:n]

#         batch["rays"] = rays[idx]          # (n, 6)
#         batch["rgb"]  = rgb[idx]           # (n, 3)
#         return batch, batch

#     # ------------------------------------------------------------------
#     # next_eval — full image for chunked rendering
#     # ------------------------------------------------------------------

#     def next_eval(self, step: int) -> Tuple[Dict, Dict]:
#         self.eval_count += 1
#         batch = _to_device(next(self._eval_iter), self.device)
#         batch["rays"] = batch["rays"].squeeze(0)
#         batch["rgb"]  = batch["rgb"].squeeze(0)
#         return batch, batch

#     def next_eval_image(self, step: int) -> Tuple[Dict, Dict]:
#         return self.next_eval(step)

#     # ------------------------------------------------------------------
#     # Convenience properties used by the pipeline
#     # ------------------------------------------------------------------

#     @property
#     def image_height(self) -> int:
#         return self._scene_data["H"]

#     @property
#     def image_width(self) -> int:
#         return self._scene_data["W"]

#     @property
#     def near(self) -> float:
#         return self._scene_data["near"]

#     @property
#     def far(self) -> float:
#         return self._scene_data["far"]

#     # ------------------------------------------------------------------
#     # Required interface stubs
#     # ------------------------------------------------------------------

#     def get_train_rays_per_batch(self) -> int:
#         return self.config.num_rays_per_batch

#     def get_eval_rays_per_batch(self) -> int:
#         return self.config.eval_chunk_size

#     def get_datapath(self):
#         from pathlib import Path
#         return Path(self.config.data_root) / self.config.scene

#     def get_param_groups(self) -> Dict[str, List[Parameter]]:
#         return {}

#     def get_training_callbacks(
#         self, attrs: TrainingCallbackAttributes
#     ) -> List[TrainingCallback]:
#         return []

#     def setup_train(self):
#         pass

#     def setup_eval(self):
#         pass


# # ---------------------------------------------------------------------------
# # Module-level helpers
# # ---------------------------------------------------------------------------

# def _collate(batch: List[Dict]) -> Dict:
#     out = {}
#     for key, val in batch[0].items():
#         vals = [b[key] for b in batch]
#         if isinstance(val, np.ndarray):
#             out[key] = torch.from_numpy(np.stack(vals))
#         elif isinstance(val, (int, float, np.integer, np.floating)):
#             out[key] = torch.tensor(vals)
#         elif isinstance(val, torch.Tensor):
#             out[key] = torch.stack(vals)
#         else:
#             out[key] = vals
#     return out


# def _to_device(batch: Dict, device) -> Dict:
#     return {
#         k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
#         for k, v in batch.items()
#     }