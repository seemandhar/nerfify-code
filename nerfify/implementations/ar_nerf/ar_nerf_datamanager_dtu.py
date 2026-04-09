# """
# AR-NeRF DataManager
# DTU-style datamanager for few-shot NeRF (3/6/9 input views on DTU and LLFF).

# Key fix: during training, we subsample `num_rays_per_batch` rays from the
# full image rather than passing all rays at once, which caused OOM.
# """

# from dataclasses import dataclass, field
# from typing import Dict, List, Literal, Tuple, Type, Union
# from itertools import cycle

# import cv2
# import imageio
# import numpy as np
# import os
# import random
# import torch
# from torch.nn import Parameter

# from nerfstudio.data.datamanagers.base_datamanager import (
#     VanillaDataManager,
#     VanillaDataManagerConfig,
# )
# from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes


# # ---------------------------------------------------------------------------
# # Utility helpers
# # ---------------------------------------------------------------------------

# def read_cam_file(cam_path: str):
#     """Read camera intrinsics and extrinsics from MVSNeRF/DTU cam file."""
#     with open(cam_path) as f:
#         lines = f.readlines()
#     extrinsic = np.array(
#         [list(map(float, lines[i + 1].split())) for i in range(3)], dtype=np.float32
#     )
#     extrinsic = np.vstack([extrinsic, [0, 0, 0, 1]]).astype(np.float32)
#     intrinsic = np.array(
#         [list(map(float, lines[i + 7].split())) for i in range(3)], dtype=np.float32
#     )
#     return intrinsic, extrinsic


# def build_rays_numpy(img: np.ndarray, ext: np.ndarray, ixt: np.ndarray) -> np.ndarray:
#     """Build all rays for an image. Returns (H*W, 6) float32 [origins | directions]."""
#     H, W = img.shape[:2]
#     fx, fy = ixt[0, 0], ixt[1, 1]
#     cx, cy = ixt[0, 2], ixt[1, 2]

#     xs, ys = np.meshgrid(
#         np.arange(W, dtype=np.float32),
#         np.arange(H, dtype=np.float32),
#         indexing="xy",
#     )
#     # Camera-space unit directions (z-forward OpenCV convention)
#     dirs = np.stack(
#         [(xs - cx) / fx, (ys - cy) / fy, np.ones_like(xs)], axis=-1
#     )  # (H, W, 3)

#     c2w    = np.linalg.inv(ext)                     # world-from-camera (4,4)
#     R, t   = c2w[:3, :3], c2w[:3, 3]
#     rays_d = (dirs @ R.T).reshape(-1, 3)            # (H*W, 3)
#     rays_o = np.broadcast_to(t, rays_d.shape).copy()

#     return np.concatenate([rays_o, rays_d], axis=-1).astype(np.float32)  # (H*W, 6)


# def read_pfm(path: str) -> np.ndarray:
#     """Read a .pfm depth file."""
#     with open(path, "rb") as f:
#         header = f.readline().decode("utf-8").rstrip()
#         dims   = f.readline().decode("utf-8")
#         W, H   = map(int, dims.split())
#         scale  = float(f.readline().decode("utf-8").rstrip())
#         data   = np.frombuffer(f.read(), dtype=np.float32).reshape(H, W)
#         if scale > 0:
#             data = np.flip(data, 0).copy()
#     return data


# # ---------------------------------------------------------------------------
# # Dataset  (stores full images; subsampling done in DataManager.next_train)
# # ---------------------------------------------------------------------------

# class ARNeRFDataset:
#     """DTU / LLFF few-shot dataset for AR-NeRF.

#     __getitem__ returns the FULL set of rays + RGB for one target view.
#     The DataManager is responsible for randomly picking `num_rays_per_batch`
#     of those rays at each training step.
#     """

#     def __init__(
#         self,
#         data_root: str,
#         ann_file: str,
#         split: str = "train",
#         input_views: int = 3,
#         depth_ranges: Tuple[float, float] = (425.0, 905.0),
#         scene: str = "",
#         pairs_path: str = "",
#     ):
#         self.data_root    = data_root
#         self.split        = split
#         self.input_views  = input_views
#         self.depth_ranges = list(depth_ranges)
#         self.scenes       = [scene] if scene else []
#         self.pairs_path   = pairs_path
#         self._build_metas(ann_file)

#     def _build_metas(self, ann_file: str):
#         scenes = [l.strip() for l in open(ann_file).readlines()]
#         if self.scenes:
#             scenes = self.scenes

#         dtu_pairs = (
#             torch.load(self.pairs_path)
#             if self.pairs_path and os.path.isfile(self.pairs_path)
#             else None
#         )

#         self.scene_infos: Dict = {}
#         self.metas: List       = []

#         for scene in scenes:
#             info: Dict = {"ixts": [], "exts": [], "dpt_paths": [], "img_paths": []}
#             for i in range(49):
#                 cam_path = os.path.join(
#                     self.data_root, "Cameras", "train", f"{i:08d}_cam.txt"
#                 )
#                 ixt, ext  = read_cam_file(cam_path)
#                 ixt[:2]  *= 4        # scale intrinsics to full resolution
#                 info["ixts"].append(ixt.astype(np.float32))
#                 info["exts"].append(ext.astype(np.float32))
#                 info["dpt_paths"].append(
#                     os.path.join(
#                         self.data_root, "Depths",
#                         f"{scene}_train", f"depth_map_{i:04d}.pfm"
#                     )
#                 )
#                 info["img_paths"].append(
#                     os.path.join(
#                         self.data_root, "Rectified",
#                         f"{scene}_train", f"rect_{i+1:03d}_3_r5000.png"
#                     )
#                 )

#             if dtu_pairs is not None:
#                 if self.split == "train" and len(self.scenes) == 1:
#                     train_ids = list(dtu_pairs["dtu_train"])
#                     test_ids  = list(dtu_pairs["dtu_train"])
#                 elif self.split == "train":
#                     train_ids = list(range(49))
#                     test_ids  = list(range(49))
#                 else:
#                     train_ids = list(dtu_pairs["dtu_train"])
#                     test_ids  = list(dtu_pairs["dtu_val"])
#             else:
#                 train_ids = test_ids = list(range(49))

#             info.update({"train_ids": train_ids, "test_ids": test_ids})
#             self.scene_infos[scene] = info

#             cam_pts = np.array(
#                 [np.linalg.inv(info["exts"][i])[:3, 3] for i in train_ids]
#             )
#             for tar_view in test_ids:
#                 cam_pt    = np.linalg.inv(info["exts"][tar_view])[:3, 3]
#                 dist      = np.linalg.norm(cam_pts - cam_pt[None], axis=-1)
#                 args      = dist.argsort()
#                 args      = args[1:] if tar_view in train_ids else args
#                 n_src     = self.input_views + 1 if self.split == "train" else self.input_views
#                 src_views = [train_ids[k] for k in args[:n_src]]
#                 self.metas.append((scene, tar_view, src_views))

#     def __len__(self):
#         return len(self.metas)

#     def __getitem__(self, index):
#         if isinstance(index, tuple):
#             index = index[0]

#         scene, tar_view, src_views = self.metas[index]
#         info = self.scene_infos[scene]

#         if self.split == "train":
#             if random.random() < 0.1:
#                 src_views = src_views + [tar_view]
#             src_views = random.sample(src_views[: self.input_views + 1], self.input_views)

#         tar_img = np.array(imageio.imread(info["img_paths"][tar_view])) / 255.0
#         H, W    = tar_img.shape[:2]
#         tar_ext = info["exts"][tar_view]
#         tar_ixt = info["ixts"][tar_view]

#         # Build ALL rays for this view
#         rays_all = build_rays_numpy(tar_img, tar_ext, tar_ixt)  # (H*W, 6)
#         rgb_all  = tar_img.reshape(-1, 3).astype(np.float32)    # (H*W, 3)

#         # Mask for eval metrics
#         if self.split != "train":
#             try:
#                 dpt      = read_pfm(info["dpt_paths"][tar_view])
#                 dpt      = cv2.resize(dpt, (W, H), interpolation=cv2.INTER_NEAREST)
#                 tar_mask = (dpt > 0.0).reshape(-1).astype(np.uint8)
#             except Exception:
#                 tar_mask = np.ones(H * W, dtype=np.uint8)
#         else:
#             tar_mask = np.ones(H * W, dtype=np.uint8)

#         src_inps, src_exts, src_ixts = self._read_src(info, src_views)

#         return {
#             "rays":     rays_all,                                   # (H*W, 6)
#             "rgb":      rgb_all,                                    # (H*W, 3)
#             "mask":     tar_mask,                                   # (H*W,)
#             "src_inps": src_inps,                                   # (N_src,3,H,W)
#             "src_exts": src_exts,                                   # (N_src,4,4)
#             "src_ixts": src_ixts,                                   # (N_src,3,3)
#             "near_far": np.array(self.depth_ranges, dtype=np.float32),
#             "H": H,
#             "W": W,
#             "meta": {"scene": scene, "tar_view": tar_view, "split": self.split},
#         }

#     def _read_src(self, info, src_views):
#         inps, exts, ixts = [], [], []
#         for sv in src_views:
#             img = np.array(imageio.imread(info["img_paths"][sv])) / 255.0
#             inps.append((img * 2.0 - 1.0).transpose(2, 0, 1))  # (3,H,W) in [-1,1]
#             exts.append(info["exts"][sv])
#             ixts.append(info["ixts"][sv])
#         return (
#             np.stack(inps).astype(np.float32),
#             np.stack(exts).astype(np.float32),
#             np.stack(ixts).astype(np.float32),
#         )


# # ---------------------------------------------------------------------------
# # DataManager
# # ---------------------------------------------------------------------------

# @dataclass
# class ARNeRFDataManagerConfig(VanillaDataManagerConfig):
#     """AR-NeRF DataManager Config."""

#     _target: Type = field(default_factory=lambda: ARNeRFDataManager)

#     # data_root:      str   = "data/dtu"
#     # ann_file_train: str   = "data/dtu/train.txt"
#     # ann_file_eval:  str   = "data/dtu/val.txt"
#     # pairs_path:     str   = ""

#     data_root: str = "/home/keshav06/ENeRF/dtu/"
#     ann_file_train: str = "/home/keshav06/ENeRF/data/mvsnerf/dtu_train_all.txt"
#     ann_file_eval: str = "/home/keshav06/ENeRF/data/mvsnerf/dtu_val_all.txt"
#     pairs_path: str = '/home/keshav06/ENeRF/data/mvsnerf/pairs.th'

#     input_views:    int   = 3
#     depth_ranges:   Tuple[float, float] = (425.0, 905.0)
#     scene:          str   = ""

#     # Number of rays sampled per training step (controls GPU memory)
#     num_rays_per_batch: int = 4096

#     # Number of rays processed at once during eval (chunked rendering)
#     eval_chunk_size: int = 4096


# class ARNeRFDataManager():
#     """AR-NeRF DataManager.

#     Training:  randomly subsamples `num_rays_per_batch` rays from the full image.
#     Evaluation: returns the full image; pipeline chunks it via `eval_chunk_size`.
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
#         # Bypass VanillaDataManager.__init__ (needs dataparser / scene_box)
#         self.config      = config
#         self.device      = device
#         self.world_size  = world_size
#         self.local_rank  = local_rank
#         self.test_mode   = test_mode
#         self.train_count = 0
#         self.eval_count  = 0

#         self.train_dataset = ARNeRFDataset(
#             data_root    = config.data_root,
#             ann_file     = config.ann_file_train,
#             split        = "train",
#             input_views  = config.input_views,
#             depth_ranges = config.depth_ranges,
#             scene        = config.scene,
#             pairs_path   = config.pairs_path,
#         )
#         self.eval_dataset = ARNeRFDataset(
#             data_root    = config.data_root,
#             ann_file     = config.ann_file_eval,
#             split        = "test",
#             input_views  = config.input_views,
#             depth_ranges = config.depth_ranges,
#             scene        = config.scene,
#             pairs_path   = config.pairs_path,
#         )

#         from torch.utils.data import DataLoader

#         def worker_init(wid):
#             np.random.seed(wid)
#             random.seed(wid)

#         self.train_loader = DataLoader(
#             self.train_dataset,
#             batch_size     = 1,
#             shuffle        = True,
#             num_workers    = 4,
#             pin_memory     = True,
#             worker_init_fn = worker_init,
#             collate_fn     = _collate_fn,
#         )
#         self.eval_loader = DataLoader(
#             self.eval_dataset,
#             batch_size  = 1,
#             shuffle     = False,
#             num_workers = 2,
#             pin_memory  = True,
#             collate_fn  = _collate_fn,
#         )

#         self.train_iter = cycle(self.train_loader)
#         self.eval_iter  = cycle(self.eval_loader)

#     # ------------------------------------------------------------------
#     # next_train  — subsample to num_rays_per_batch
#     # ------------------------------------------------------------------

#     def next_train(self, step: int) -> Tuple[Dict, Dict]:
#         """Fetch one image then pick `num_rays_per_batch` random rays from it.

#         This is the critical fix for OOM: instead of forwarding all H*W rays
#         (e.g. 640*512 = 327 680) we only forward 4 096 at a time.
#         """
#         self.train_count += 1
#         batch = _to_device(next(self.train_iter), self.device)

#         rays = batch["rays"].squeeze(0)   # (H*W, 6)
#         rgb  = batch["rgb"].squeeze(0)    # (H*W, 3)

#         N = rays.shape[0]
#         n = min(self.config.num_rays_per_batch, N)

#         # Random permutation — fast on GPU/CPU
#         idx = torch.randperm(N, device=self.device)[:n]

#         batch["rays"] = rays[idx]   # (n, 6)
#         batch["rgb"]  = rgb[idx]    # (n, 3)

#         return batch, batch

#     # ------------------------------------------------------------------
#     # next_eval — return full image (pipeline will chunk)
#     # ------------------------------------------------------------------

#     def next_eval(self, step: int) -> Tuple[Dict, Dict]:
#         """Full image for evaluation. Pipeline renders in chunks of `eval_chunk_size`."""
#         self.eval_count += 1
#         batch = _to_device(next(self.eval_iter), self.device)
#         batch["rays"] = batch["rays"].squeeze(0)   # (H*W, 6)
#         batch["rgb"]  = batch["rgb"].squeeze(0)    # (H*W, 3)
#         return batch, batch

#     def next_eval_image(self, step: int) -> Tuple[Dict, Dict]:
#         return self.next_eval(step)

#     # ------------------------------------------------------------------
#     # Required interface stubs
#     # ------------------------------------------------------------------

#     def get_train_rays_per_batch(self) -> int:
#         return self.config.num_rays_per_batch

#     def get_eval_rays_per_batch(self) -> int:
#         return self.config.eval_chunk_size

#     def get_datapath(self):
#         from pathlib import Path
#         return Path(self.config.data_root)

#     def get_param_groups(self) -> Dict[str, List[Parameter]]:
#         return {}

#     def get_training_callbacks(self, attrs: TrainingCallbackAttributes) -> List[TrainingCallback]:
#         return []

#     def setup_train(self):
#         pass

#     def setup_eval(self):
#         pass


# # ---------------------------------------------------------------------------
# # Module-level helpers
# # ---------------------------------------------------------------------------

# def _collate_fn(batch):
#     """Collate list of dataset items, converting numpy arrays to tensors."""
#     elem = batch[0]
#     out  = {}
#     for key, val in elem.items():
#         if key == "meta":
#             out[key] = val
#             continue
#         if isinstance(val, np.ndarray):
#             out[key] = torch.from_numpy(np.stack([b[key] for b in batch]))
#         elif isinstance(val, (int, float)):
#             out[key] = torch.tensor([b[key] for b in batch])
#         elif isinstance(val, torch.Tensor):
#             out[key] = torch.stack([b[key] for b in batch])
#         else:
#             out[key] = [b[key] for b in batch]
#     return out


# def _to_device(batch: Dict, device) -> Dict:
#     return {
#         k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
#         for k, v in batch.items()
#     }

"""
AR-NeRF DataManager — per-scene optimisation.

No target/source view distinction. We load ALL images for a single scene,
randomly sample `num_rays_per_batch` rays from a randomly chosen image each
training step, and use a held-out subset for evaluation.

Usage (nerfstudio CLI override example):
    arnerf_method --pipeline.datamanager.data_root data/dtu \
                  --pipeline.datamanager.scene scan114 \
                  --pipeline.datamanager.input_views 3
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import cycle
from typing import Dict, List, Literal, Tuple, Type, Union

import cv2
import imageio
import numpy as np
import os
import random
import torch
from torch.nn import Parameter
from torch.utils.data import DataLoader, Dataset

from nerfstudio.data.datamanagers.base_datamanager import (
    VanillaDataManager,
    VanillaDataManagerConfig,
)
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes


# ---------------------------------------------------------------------------
# Camera file reader (MVSNeRF / DTU format)
# ---------------------------------------------------------------------------

def _read_cam_file(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Return (intrinsic 3×3, extrinsic 4×4) from a DTU cam file."""
    with open(path) as f:
        lines = f.readlines()
    ext = np.array(
        [list(map(float, lines[i + 1].split())) for i in range(3)],
        dtype=np.float32,
    )
    ext = np.vstack([ext, [0, 0, 0, 1]]).astype(np.float32)
    ixt = np.array(
        [list(map(float, lines[i + 7].split())) for i in range(3)],
        dtype=np.float32,
    )
    return ixt, ext


# ---------------------------------------------------------------------------
# Ray builder
# ---------------------------------------------------------------------------

def _build_rays(img: np.ndarray, ext: np.ndarray, ixt: np.ndarray) -> np.ndarray:
    """Return (H*W, 6) float32 ray array [ox,oy,oz, dx,dy,dz]."""
    H, W = img.shape[:2]
    fx, fy = ixt[0, 0], ixt[1, 1]
    cx, cy = ixt[0, 2], ixt[1, 2]

    xs, ys = np.meshgrid(
        np.arange(W, dtype=np.float32),
        np.arange(H, dtype=np.float32),
        indexing="xy",
    )
    # Camera-space directions (OpenCV / z-forward)
    dirs = np.stack([(xs - cx) / fx, (ys - cy) / fy, np.ones_like(xs)], axis=-1)

    c2w    = np.linalg.inv(ext)
    rays_d = (dirs @ c2w[:3, :3].T).reshape(-1, 3)
    rays_o = np.broadcast_to(c2w[:3, 3], rays_d.shape).copy()
    return np.concatenate([rays_o, rays_d], axis=-1).astype(np.float32)


# ---------------------------------------------------------------------------
# Per-scene image store
# ---------------------------------------------------------------------------

class SceneImageStore:
    """Loads and caches all images + cameras for a single scene.

    Args:
        data_root:    Root of the DTU dataset.
        scene:        Scene name, e.g. 'scan114'.
        (scans for evaluating -  8, 21, 30, 31, 34, 38, 40, 41, 45, 55, 63, 82, 103, 110, 114)
        input_views:  How many views to use for training (sampled randomly).
                      The rest are held out for evaluation.
        depth_ranges: (near, far) depth bounds.
        pairs_path:   Optional path to dtu_pairs.th for canonical splits.
        scale:        Intrinsic scale factor (default 4 for DTU full-res).
    """

    def __init__(
        self,
        data_root: str,
        scene: str,
        input_views: int = 3,
        depth_ranges: Tuple[float, float] = (425.0, 905.0),
        pairs_path: str = "",
        scale: float = 4.0,
    ):
        self.data_root    = data_root
        self.scene        = scene
        self.input_views  = input_views
        self.near, self.far = depth_ranges

        # ---- collect all 49 cameras / images for this scene ----
        all_imgs, all_rays, all_ixts, all_exts = [], [], [], []

        for i in range(49):
            cam_path = os.path.join(
                data_root, "Cameras", "train", f"{i:08d}_cam.txt"
            )
            img_path = os.path.join(
                data_root, "Rectified", f"{scene}_train",
                f"rect_{i+1:03d}_3_r5000.png",
            )

            ixt, ext = _read_cam_file(cam_path)
            ixt[:2] *= scale                           # scale to full resolution

            img = np.array(imageio.imread(img_path), dtype=np.float32) / 255.0
            rays = _build_rays(img, ext, ixt)          # (H*W, 6)

            all_imgs.append(img)
            all_rays.append(rays)
            all_ixts.append(ixt)
            all_exts.append(ext)

        self.all_imgs  = all_imgs   # list of (H, W, 3) float32
        self.all_rays  = all_rays   # list of (H*W, 6)  float32
        self.all_ixts  = all_ixts
        self.all_exts  = all_exts
        self.H, self.W = all_imgs[0].shape[:2]
        self.n_total   = len(all_imgs)  # 49

        # ---- decide train / eval split ----
        if pairs_path and os.path.isfile(pairs_path):
            pairs = torch.load(pairs_path)
            self.train_ids = list(pairs["dtu_train"])   # canonical 28 views
            self.eval_ids  = list(pairs["dtu_val"])     # canonical 21 views
        else:
            # Random split: first `input_views` indices for training,
            # remainder for eval.  Shuffle with a fixed seed for reproducibility.
            rng = np.random.default_rng(42)
            perm = rng.permutation(self.n_total).tolist()
            self.train_ids = perm[:input_views]
            self.eval_ids  = perm[input_views:]

        # For few-shot setting restrict training to exactly `input_views` images
        # (randomly sampled from the canonical train set).
        rng2 = np.random.default_rng(0)
        self.train_ids = list(
            rng2.choice(self.train_ids, size=min(input_views, len(self.train_ids)),
                        replace=False)
        )

        print(
            f"[ARNeRF] Scene '{scene}': "
            f"{len(self.train_ids)} train views {self.train_ids}, "
            f"{len(self.eval_ids)} eval views."
        )


# ---------------------------------------------------------------------------
# Torch Datasets
# ---------------------------------------------------------------------------

class ARNeRFTrainDataset(Dataset):
    """Wraps SceneImageStore for training.

    Each item is ALL rays + RGB for one randomly selected training view.
    The DataManager subsamples `num_rays_per_batch` rays from it.
    """

    def __init__(self, store: SceneImageStore):
        self.store = store

    def __len__(self):
        # Return a large number so DataLoader never runs out.
        # Actual epoch semantics don't matter for NeRF per-scene training.
        return len(self.store.train_ids) * 1000

    def __getitem__(self, idx: int) -> Dict:
        # Pick one training view uniformly at random
        view_id = random.choice(self.store.train_ids)
        rays    = self.store.all_rays[view_id]          # (H*W, 6)
        rgb     = self.store.all_imgs[view_id].reshape(-1, 3)  # (H*W, 3)
        return {
            "rays":     rays.astype(np.float32),
            "rgb":      rgb.astype(np.float32),
            "near_far": np.array([self.store.near, self.store.far], dtype=np.float32),
            "view_id":  np.int32(view_id),
        }


class ARNeRFEvalDataset(Dataset):
    """Wraps SceneImageStore for evaluation (full images)."""

    def __init__(self, store: SceneImageStore):
        self.store = store

    def __len__(self):
        return len(self.store.eval_ids)

    def __getitem__(self, idx: int) -> Dict:
        view_id = self.store.eval_ids[idx]
        rays    = self.store.all_rays[view_id]
        rgb     = self.store.all_imgs[view_id].reshape(-1, 3)
        return {
            "rays":     rays.astype(np.float32),
            "rgb":      rgb.astype(np.float32),
            "near_far": np.array([self.store.near, self.store.far], dtype=np.float32),
            "view_id":  np.int32(view_id),
            "H":        np.int32(self.store.H),
            "W":        np.int32(self.store.W),
        }


# ---------------------------------------------------------------------------
# DataManager
# ---------------------------------------------------------------------------

@dataclass
class ARNeRFDataManagerConfig(VanillaDataManagerConfig):
    """AR-NeRF per-scene DataManager config."""

    _target: Type = field(default_factory=lambda: ARNeRFDataManager)

    # ---- scene specification ----
    data_root: str = "/home/keshav06/ENeRF/dtu/"
    """Root directory of the DTU dataset."""
    scene: str = "scan8"
    """Scene name to optimise (e.g. 'scan114')."""
    pairs_path: str = '/home/keshav06/ENeRF/data/mvsnerf/pairs.th'
    """Optional path to dtu pairs.th for canonical train/eval split."""
    input_views: int = 3
    """Number of training views (3, 6, or 9 for DTU few-shot setting)."""
    depth_ranges: Tuple[float, float] = (425.0, 905.0)
    """Near/far depth bounds in scene units."""
    ixt_scale: float = 4.0
    """Intrinsic scale factor (4 for DTU full-resolution images)."""



    # ---- ray sampling ----
    num_rays_per_batch: int = 4096
    """Rays sampled per training step — the main knob for GPU memory."""
    eval_chunk_size: int = 4096
    """Rays per forward pass during evaluation (chunked rendering)."""


class ARNeRFDataManager():
    """AR-NeRF DataManager — per-scene optimisation.

    * Loads ALL images for the specified scene once at startup.
    * Training: each step picks one random training view, then randomly
      subsamples `num_rays_per_batch` rays from it.
    * Evaluation: full images rendered in `eval_chunk_size` chunks by
      the pipeline.
    """

    config: ARNeRFDataManagerConfig

    def __init__(
        self,
        config: ARNeRFDataManagerConfig,
        device: Union[torch.device, str] = "cpu",
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        **kwargs,
    ):
        # Bypass VanillaDataManager.__init__ (requires a nerfstudio dataparser)
        self.config      = config
        self.device      = device
        self.world_size  = world_size
        self.local_rank  = local_rank
        self.test_mode   = test_mode
        self.train_count = 0
        self.eval_count  = 0

        # Load scene once
        self._store = SceneImageStore(
            data_root    = config.data_root,
            scene        = config.scene,
            input_views  = config.input_views,
            depth_ranges = config.depth_ranges,
            pairs_path   = config.pairs_path,
            scale        = config.ixt_scale,
        )

        self.train_dataset = ARNeRFTrainDataset(self._store)
        self.eval_dataset  = ARNeRFEvalDataset(self._store)

        def _worker_init(wid):
            np.random.seed(wid)
            random.seed(wid)

        self._train_loader = DataLoader(
            self.train_dataset,
            batch_size     = 1,
            shuffle        = True,
            num_workers    = 2,
            pin_memory     = True,
            worker_init_fn = _worker_init,
            collate_fn     = _collate,
        )
        self._eval_loader = DataLoader(
            self.eval_dataset,
            batch_size  = 1,
            shuffle     = False,
            num_workers = 2,
            pin_memory  = True,
            collate_fn  = _collate,
        )

        self._train_iter = cycle(self._train_loader)
        self._eval_iter  = cycle(self._eval_loader)

    # ------------------------------------------------------------------
    # next_train — subsample num_rays_per_batch rays from one view
    # ------------------------------------------------------------------

    def next_train(self, step: int) -> Tuple[Dict, Dict]:
        """Return a random subset of rays from a randomly chosen training view.

        Memory cost is O(num_rays_per_batch), not O(H*W).
        """
        self.train_count += 1
        batch = _to_device(next(self._train_iter), self.device)

        rays = batch["rays"].squeeze(0)    # (H*W, 6)
        rgb  = batch["rgb"].squeeze(0)     # (H*W, 3)

        n   = min(self.config.num_rays_per_batch, rays.shape[0])
        idx = torch.randperm(rays.shape[0], device=self.device)[:n]

        batch["rays"] = rays[idx]          # (n, 6)
        batch["rgb"]  = rgb[idx]           # (n, 3)
        return batch, batch

    # ------------------------------------------------------------------
    # next_eval — full image (pipeline chunks it)
    # ------------------------------------------------------------------

    def next_eval(self, step: int) -> Tuple[Dict, Dict]:
        self.eval_count += 1
        batch = _to_device(next(self._eval_iter), self.device)
        batch["rays"] = batch["rays"].squeeze(0)   # (H*W, 6)
        batch["rgb"]  = batch["rgb"].squeeze(0)    # (H*W, 3)
        return batch, batch

    def next_eval_image(self, step: int) -> Tuple[Dict, Dict]:
        return self.next_eval(step)

    # ------------------------------------------------------------------
    # Convenience: access store metadata in the pipeline
    # ------------------------------------------------------------------

    @property
    def image_height(self) -> int:
        return self._store.H

    @property
    def image_width(self) -> int:
        return self._store.W

    # ------------------------------------------------------------------
    # Required interface stubs
    # ------------------------------------------------------------------

    def get_train_rays_per_batch(self) -> int:
        return self.config.num_rays_per_batch

    def get_eval_rays_per_batch(self) -> int:
        return self.config.eval_chunk_size

    def get_datapath(self):
        from pathlib import Path
        return Path(self.config.data_root) / self.config.scene

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        return {}

    def get_training_callbacks(
        self, attrs: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        return []

    def setup_train(self):
        pass

    def setup_eval(self):
        pass


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _collate(batch: List[Dict]) -> Dict:
    """Collate list of dicts — numpy arrays become tensors, others stacked."""
    out = {}
    for key, val in batch[0].items():
        vals = [b[key] for b in batch]
        if isinstance(val, np.ndarray):
            out[key] = torch.from_numpy(np.stack(vals))
        elif isinstance(val, (int, float, np.integer, np.floating)):
            out[key] = torch.tensor(vals)
        elif isinstance(val, torch.Tensor):
            out[key] = torch.stack(vals)
        else:
            out[key] = vals
    return out


def _to_device(batch: Dict, device) -> Dict:
    return {
        k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }