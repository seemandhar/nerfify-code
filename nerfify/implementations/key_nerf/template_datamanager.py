"""
Template DataManager with informative view and ray selection.

Implements:
- Greedy camera subset selection maximizing baseline diversity (pairwise optical-axis angle).
- Per-image informativeness map (default: gradient magnitude pooled over a local window).
- Mixed sampling: entropy-biased rays (B * frac) + uniform rays (B * (1-frac)) from a selected camera.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Set, Tuple, Type, Union
from scipy.optimize import linprog
from nerfstudio.cameras.cameras import Cameras
import math
import random
import numpy as np
import torch
import torch.nn.functional as F
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.datamanagers.base_datamanager import (
    VanillaDataManager,
    VanillaDataManagerConfig,
)


@dataclass
class TemplateDataManagerConfig(VanillaDataManagerConfig):
    num_selected_cameras: int = 16
    use_view_selection: bool = True
    entropy_mix_frac: float = 0.5
    entropy_window: int = 9
    entropy_use_gradient: bool = True
    seed: int = 42

    _target: Type = field(default_factory=lambda: TemplateDataManager)


class ILPCameraSelector:
    
    def __init__(
        self,
        grid_resolution: int = 32,
        bounds_padding: float = 0.1,
        use_relaxation: bool = True,
    ):
        self.grid_resolution = grid_resolution
        self.bounds_padding = bounds_padding
        self.use_relaxation = use_relaxation
        
    def compute_scene_bounds(self,cameras):
        camera_centers = cameras.camera_to_worlds[:, :3, 3]  # (N, 3)
        
        pmin = camera_centers.min(dim=0)[0]
        pmax = camera_centers.max(dim=0)[0]
        
        center = (pmin + pmax) / 2
        extent = pmax - pmin
        pmin = center - extent * (0.5 + self.bounds_padding)
        pmax = center + extent * (0.5 + self.bounds_padding)
        
        return pmin, pmax
    
    def create_grid_points(self,pmin,pmax):
        res = self.grid_resolution
        
        x = torch.linspace(pmin[0], pmax[0], res)
        y = torch.linspace(pmin[1], pmax[1], res)
        z = torch.linspace(pmin[2], pmax[2], res)
        
        grid_x, grid_y, grid_z = torch.meshgrid(x, y, z, indexing='ij')
        
        grid_points = torch.stack([
            grid_x.flatten(),
            grid_y.flatten(),
            grid_z.flatten()
        ], dim=-1)  # (M, 3)
        
        return grid_points
    
    def compute_visibility_matrix(self,cameras,grid_points):
        N = len(cameras)
        M = len(grid_points)
        A = np.zeros((N, M), dtype=np.float64)
        
        points_homo = torch.cat([
            grid_points,
            torch.ones(M, 1, device=grid_points.device)
        ], dim=-1)  # (M, 4)
        
        for i in range(N):
            c2w = cameras.camera_to_worlds[i]  # (4, 4) or (3, 4)
            if c2w.shape[0] == 3:
                c2w = torch.cat([c2w, torch.tensor([[0, 0, 0, 1]], device=c2w.device)], dim=0)
            
            w2c = torch.inverse(c2w)
            
            points_cam = (w2c @ points_homo.T).T  # (M, 4)
            points_cam = points_cam[:, :3]  # (M, 3)
            
            in_front = points_cam[:, 2] > 0
            
            K = cameras.get_intrinsics_matrices()[i]  # (3, 3)
            points_proj = (K @ points_cam.T).T  # (M, 3)
            points_proj = points_proj[:, :2] / (points_proj[:, 2:3] + 1e-8)  # (M, 2)
            
            height = cameras.height[i] if cameras.height.numel() > 1 else cameras.height
            width = cameras.width[i] if cameras.width.numel() > 1 else cameras.width
            
            in_bounds = (
                (points_proj[:, 0] >= 0) &
                (points_proj[:, 0] < width) &
                (points_proj[:, 1] >= 0) &
                (points_proj[:, 1] < height)
            )
            
            visible = in_front & in_bounds
            A[i, :] = visible.cpu().numpy().astype(np.float64)
        
        return A
    
    def solve_ilp(self,A):
        N, M = A.shape
        
        c = np.ones(N)
        
        A_ub = -A.T
        b_ub = -np.ones(M)

        bounds = [(0, 1) for _ in range(N)]
        
        if self.use_relaxation:
            result = linprog(
                c=c,
                A_ub=A_ub,
                b_ub=b_ub,
                bounds=bounds,
                method='highs'
            )
            
            if not result.success:
                raise ValueError("LP optimization failed")
            
            x = result.x
            selected = np.where(x > 0.5)[0].tolist()
            
            coverage = (A[selected, :].sum(axis=0) > 0)
            uncovered = np.where(~coverage)[0]
            
            while len(uncovered) > 0:
                remaining_cams = [i for i in range(N) if i not in selected]
                if not remaining_cams:
                    break
                    
                coverage_counts = [A[i, uncovered].sum() for i in remaining_cams]
                best_cam = remaining_cams[np.argmax(coverage_counts)]
                selected.append(best_cam)
                
                coverage = (A[selected, :].sum(axis=0) > 0)
                uncovered = np.where(~coverage)[0]
        else:
            selected = []
            covered = np.zeros(M, dtype=bool)
            
            while not covered.all():
                remaining = [i for i in range(N) if i not in selected]
                if not remaining:
                    break
                    
                uncovered_mask = ~covered
                coverage_counts = [A[i, uncovered_mask].sum() for i in remaining]
                best_cam = remaining[np.argmax(coverage_counts)]
                
                selected.append(best_cam)
                covered |= (A[best_cam, :] > 0)
        
        return sorted(selected)
    
    def select_cameras(self,cameras: Cameras,) -> List[int]:
        pmin, pmax = self.compute_scene_bounds(cameras)
        
        grid_points = self.create_grid_points(pmin, pmax)
        
        A = self.compute_visibility_matrix(cameras, grid_points)
        
        visible_points = A.sum(axis=0) > 0
        if not visible_points.all():
            print(f"Warning: {(~visible_points).sum()} points not visible by any camera")
            A = A[:, visible_points]
        
        selected_indices = self.solve_ilp(A)
        
        print(f"Selected {len(selected_indices)} cameras out of {len(cameras)}")
        
        return selected_indices


class TemplateDataManager(VanillaDataManager):

    config: TemplateDataManagerConfig

    _selected_cameras: Optional[List[int]]
    _selected_cameras_set: Set[int]
    _entropy_cache: Dict[int, torch.Tensor]

    def __init__(
        self,
        config: TemplateDataManagerConfig,
        device: Union[torch.device, str] = "cuda",
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        **kwargs,
    ):
        # Seed
        random.seed(config.seed)
        torch.manual_seed(config.seed)
        super().__init__(
            config=config, device=device, test_mode=test_mode, world_size=world_size, local_rank=local_rank, **kwargs
        )
        self._selected_cameras = None
        self._selected_cameras_set = set()
        self._entropy_cache = {}
        selector = ILPCameraSelector(
            grid_resolution=2,
            bounds_padding=0.1,
            use_relaxation=True,
        )
        
        self.selected_indices = selector.select_cameras(self.train_dataset.cameras)
        print(self.selected_indices)

        if self.train_dataset is not None and self.config.use_view_selection:
            self._selected_cameras = self._greedy_select_cameras(
                num_select=min(self.config.num_selected_cameras, len(self.train_dataset))
            )
            self._selected_cameras_set = set(self._selected_cameras)
        else:
            self._selected_cameras = None
            self._selected_cameras_set = set()

        print(self._selected_cameras_set)

    def _get_camera_axes(self) -> torch.Tensor:
        assert self.train_dataset is not None
        cams = self.train_dataset.cameras
        assert cams is not None and cams.camera_to_worlds is not None
        c2w = cams.camera_to_worlds.to("cpu")  # (N, 4, 4)
        axes = c2w[:, :3, 2]
        axes = F.normalize(axes, dim=-1)
        return axes  # (N,3)

    def _greedy_select_cameras(self, num_select: int) -> List[int]:
        """Farthest-point sampling on sphere using optical-axis baseline angles."""
        N = len(self.train_dataset) if self.train_dataset is not None else 0
        if N == 0 or num_select >= N:
            return list(range(N))
        axes = self._get_camera_axes()  # (N,3)
        dots = torch.clamp(axes @ axes.t(), -1.0, 1.0)
        angles = torch.arccos(dots)
        current: List[int] = self.selected_indices
        remaining: Set[int] = set([i for i in range(0, N) if i not in self.selected_indices])
        while len(current) < num_select and remaining:
            cand = list(remaining)
            sub = angles[torch.tensor(cand, dtype=torch.long), :][:, torch.tensor(current, dtype=torch.long)]
            min_to_set = torch.min(sub, dim=1).values
            pick_idx = int(torch.argmax(min_to_set).item())
            picked = cand[pick_idx]
            current.append(picked)
            remaining.remove(picked)
        return current

    @staticmethod
    def _to_grayscale(image: torch.Tensor) -> torch.Tensor:
        if image.shape[-1] == 4:
            image = image[..., :3] * image[..., 3:3+1] + (1 - image[..., 3:3+1]) * 1.0
        rgb = image[..., :3]
        gray = 0.2989 * rgb[..., 0] + 0.5870 * rgb[..., 1] + 0.1140 * rgb[..., 2]
        return gray[None, ...]  # [1,H,W]

    def _informativeness_map(self, img: torch.Tensor, win: int, use_grad: bool) -> torch.Tensor:
        with torch.no_grad():
            gray = self._to_grayscale(img)  # [1,H,W]
            gray = gray.to(self.device)
            if use_grad:
                kx = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32, device=self.device)
                ky = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32, device=self.device)
                kx = kx.view(1, 1, 3, 3)
                ky = ky.view(1, 1, 3, 3)
                gx = F.conv2d(gray, kx, padding=1)
                gy = F.conv2d(gray, ky, padding=1)
                mag = torch.sqrt(gx * gx + gy * gy)  # [1,H,W]
                pool = F.avg_pool2d(mag, kernel_size=win, stride=1, padding=win // 2)
                score = pool[0]  # [H,W]
            else:
                mean = F.avg_pool2d(gray, kernel_size=win, stride=1, padding=win // 2)
                mean2 = F.avg_pool2d(gray * gray, kernel_size=win, stride=1, padding=win // 2)
                var = torch.clamp(mean2 - mean * mean, min=0.0)
                score = var[0]
            score = torch.clamp(score, min=1e-8)
            score = score.to(torch.float32) / torch.sum(score)
            return score  # [H,W]

    def _get_or_build_entropy(self, image_idx: int, image_tensor: torch.Tensor) -> torch.Tensor:
        if image_idx in self._entropy_cache:
            return self._entropy_cache[image_idx]
        H, W = image_tensor.shape[0], image_tensor.shape[1]
        prob_map = self._informativeness_map(
            image_tensor, win=max(3, int(self.config.entropy_window) | 1), use_grad=self.config.entropy_use_gradient
        )
        prob_flat = prob_map.reshape(-1)
        self._entropy_cache[image_idx] = prob_flat
        return prob_flat

    def _choose_image_from_batch(self, image_batch: Dict) -> Tuple[int, torch.Tensor]:
        img_idx_batch = image_batch.get("image_idx", None)
        img_idx_batch = image_batch.get("image_idx", None)
        images = image_batch["image"]
        if img_idx_batch is None:
            return 0, images[0]
        img_idx_batch = img_idx_batch.tolist() if torch.is_tensor(img_idx_batch) else list(img_idx_batch)
        candidate_ids: List[int]
        if self._selected_cameras is not None and len(self._selected_cameras_set) > 0:
            candidate_ids = [i for i in img_idx_batch if i in self._selected_cameras_set]
        else:
            candidate_ids = img_idx_batch
        if len(candidate_ids) == 0:
            pick = int(img_idx_batch[0])
        else:
            pick = int(random.choice(candidate_ids))
        where = img_idx_batch.index(pick)
        return pick, images[where]

    def next_train(self, step: int) -> Tuple[RayBundle, Dict]:
        # return super().next_train(step)
        self.train_count += 1

        while True:
            image_batch = next(self.iter_train_image_dataloader)
            assert isinstance(image_batch, dict)
            pick_idx, image_tensor = self._choose_image_from_batch(image_batch)
            if (not self._selected_cameras_set) or (pick_idx in self._selected_cameras_set):
                break

        prob_flat = self._get_or_build_entropy(pick_idx, image_tensor)  # [H*W]
        H, W = image_tensor.shape[0], image_tensor.shape[1]
        total_rays = int(self.config.train_num_rays_per_batch)
        num_entropy = int(math.floor(total_rays * float(self.config.entropy_mix_frac)))
        num_uniform = max(0, total_rays - num_entropy)

        idx_entropy = torch.multinomial(prob_flat, num_samples=num_entropy, replacement=True) if num_entropy > 0 else torch.empty((0,), dtype=torch.long, device=prob_flat.device)
        idx_uniform = torch.randint(low=0, high=H * W, size=(num_uniform,), device=prob_flat.device, dtype=torch.long) if num_uniform > 0 else torch.empty((0,), dtype=torch.long, device=prob_flat.device)
        flat_indices = torch.cat([idx_entropy, idx_uniform], dim=0)  # [B]
        perm = torch.randperm(flat_indices.shape[0], device=flat_indices.device)
        flat_indices = flat_indices[perm]

        ys = torch.div(flat_indices, W, rounding_mode="floor")
        xs = flat_indices % W
        img_ids = torch.full_like(xs, fill_value=pick_idx)
        indices = torch.stack([img_ids, ys, xs], dim=-1).to(self.device)  # [B,3], long

        colors = image_tensor.to(self.device)[ys, xs, :3]  # [B,3]


        batch = {
            "indices": indices,
            "image": colors,
        }
        ray_bundle = self.train_ray_generator(indices.to(self.train_ray_generator.image_coords.device))
        return ray_bundle, batch
