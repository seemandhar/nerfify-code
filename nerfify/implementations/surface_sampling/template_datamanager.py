"""
Template DataManager
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Literal, Tuple, Type, Union, Optional, List

import torch
import torch.nn.functional as F
from nerfstudio.cameras.cameras import Cameras
from nerfify.methods.surface_sampling.template_dataparser import BlenderWithDepthDataParserConfig
from nerfstudio.configs.dataparser_configs import AnnotatedDataParserUnion
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.datamanagers.base_datamanager import (
    VanillaDataManager,
    VanillaDataManagerConfig,
)


@dataclass
class TemplateDataManagerConfig(VanillaDataManagerConfig):
    """Template DataManager Config (add paper-specific sampling switches here if needed)."""
    _target: Type = field(default_factory=lambda: TemplateDataManager)
    dataparser: AnnotatedDataParserUnion = field(default_factory=BlenderWithDepthDataParserConfig)
    # Near-surface sampling
    enable_near_surface_sampling: bool = True
    near_surface_alpha: float = 1.0 / 8.0
    ns_num_samples: int = 64
    jitter_near_surface: bool = True

    # Test-time depth from offline point cloud
    use_pointcloud_for_eval: bool = True
    pc_offline_num_views: int = 20
    pc_stride: int = 4
    pc_tau: float = 0.1

    # Hole filling
    depth_hole_fill_kappa: float = 2.0
    depth_hole_fill_window: int = 11


class TemplateDataManager(VanillaDataManager):
    """Thin wrapper around VanillaDataManager with near-surface sampling and offline point cloud."""

    config: TemplateDataManagerConfig

    def __init__(
        self,
        config: TemplateDataManagerConfig,
        device: Union[torch.device, str] = "cpu",
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        **kwargs,
    ):
        super().__init__(
            config=config, device=device, test_mode=test_mode, world_size=world_size, local_rank=local_rank, **kwargs
        )
        # print(self.)
        self._pointcloud_world: Optional[torch.Tensor] = None  # (N,3) on CPU
        self._depth_cache: Dict[int, torch.Tensor] = {}  # camera_index -> depth map HxW (CPU)
        # Build point cloud offline if requested and datasets available
        if self.config.use_pointcloud_for_eval and self.train_dataset is not None:
            try:
                self._build_offline_pointcloud()
            except Exception:
                # Fail safe: proceed without point cloud
                self._pointcloud_world = None
                self._depth_cache = {}

    # =========================
    # Ray utilities
    # =========================

    @staticmethod
    def _get_camera_hw(cameras: Cameras, cam_idx: int) -> Tuple[int, int]:
        H = int(cameras.height[cam_idx]) if isinstance(cameras.height, torch.Tensor) else int(cameras.height)
        W = int(cameras.width[cam_idx]) if isinstance(cameras.width, torch.Tensor) else int(cameras.width)
        return H, W

    @staticmethod
    def _get_intrinsics(cameras: Cameras, cam_idx: int) -> Tuple[float, float, float, float]:
        # fx, fy, cx, cy may be tensors per camera
        fx = float(cameras.fx[cam_idx]) if isinstance(cameras.fx, torch.Tensor) else float(cameras.fx)
        fy = float(cameras.fy[cam_idx]) if isinstance(cameras.fy, torch.Tensor) else float(cameras.fy)
        cx = float(cameras.cx[cam_idx]) if isinstance(cameras.cx, torch.Tensor) else float(cameras.cx)
        cy = float(cameras.cy[cam_idx]) if isinstance(cameras.cy, torch.Tensor) else float(cameras.cy)
        return fx, fy, cx, cy

    @staticmethod
    def _get_Tcw(cameras: Cameras, cam_idx: int) -> torch.Tensor:
        # (4,4) cam_to_world matrix for camera index
        Tcw = cameras.camera_to_worlds[cam_idx]
        if Tcw.shape == (3, 4):
            # Convert to 4x4
            pad = torch.tensor([[0, 0, 0, 1.0]], dtype=Tcw.dtype, device=Tcw.device)
            Tcw = torch.cat([Tcw, pad], dim=0)
        return Tcw

    def _backproject_depth_to_points(
        self, depth: torch.Tensor, cam_idx: int, stride: int = 1
    ) -> torch.Tensor:
        """Backproject a depth image (HxW) to world-space points with stride subsampling. Returns (N,3) on CPU."""
        assert self.train_dataset is not None
        cameras = self.train_dataset.cameras
        H, W = self._get_camera_hw(cameras, cam_idx)
        fx, fy, cx, cy = self._get_intrinsics(cameras, cam_idx)
        device = depth.device
        ys = torch.arange(0, H, stride, device=device)
        xs = torch.arange(0, W, stride, device=device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        z = depth[grid_y.long(), grid_x.long()]  # (h', w')
        mask = z > 0
        grid_x = grid_x[mask].float()
        grid_y = grid_y[mask].float()
        z = z[mask].float()
        if z.numel() == 0:
            return torch.empty((0, 3), dtype=torch.float32)
        x_cam = (grid_x - cx) / fx * z
        y_cam = (grid_y - cy) / fy * z
        ones = torch.ones_like(z)
        P_cam = torch.stack([x_cam, y_cam, z, ones], dim=-1).T  # (4,N)
        Tcw = self._get_Tcw(cameras, cam_idx).to(device)
        Pw = (Tcw @ P_cam).T[:, :3]  # (N,3)
        return Pw.detach().cpu()

    def _project_points_to_camera(
        self, points_world: torch.Tensor, cam_idx: int, H: int, W: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project world points to pixels (u,v) and depths z_cam for a given camera.
        Returns (u, v, z) each shape (N,), where u in [0,W-1], v in [0,H-1].
        Filters points with z<=0 or out of image bounds."""
        assert self.train_dataset is not None
        cameras = self.train_dataset.cameras
        fx, fy, cx, cy = self._get_intrinsics(cameras, cam_idx)
        Tcw = self._get_Tcw(cameras, cam_idx).detach().cpu()
        Twc = torch.inverse(Tcw)
        # World to camera: X_cam = Twc^{-1} * [X_w,1]
        ones = torch.ones((points_world.shape[0], 1), dtype=points_world.dtype)
        Pw_h = torch.cat([points_world, ones], dim=-1)  # (N,4)
        Pc_h = (Twc.inverse() @ Pw_h.T).T  # (N,4)
        Xc = Pc_h[:, 0]
        Yc = Pc_h[:, 1]
        Zc = Pc_h[:, 2]
        valid = Zc > 1e-6
        u = fx * (Xc[valid] / Zc[valid]) + cx
        v = fy * (Yc[valid] / Zc[valid]) + cy
        z = Zc[valid]
        # Filter to image bounds
        u_round = torch.round(u).long()
        v_round = torch.round(v).long()
        in_bounds = (u_round >= 0) & (u_round < W) & (v_round >= 0) & (v_round < H)
        return u_round[in_bounds], v_round[in_bounds], z[in_bounds]

    def _depth_from_pointcloud_for_camera(self, cam_idx: int) -> Optional[torch.Tensor]:
        """Project point cloud and build a per-pixel depth map (H,W) with nearest depth; returns CPU tensor."""
        if self._pointcloud_world is None or self.train_dataset is None:
            return None
        cameras = self.train_dataset.cameras
        H, W = self._get_camera_hw(cameras, cam_idx)
        depth = torch.zeros((H, W), dtype=torch.float32)
        if self._pointcloud_world.numel() == 0:
            return depth
        u, v, z = self._project_points_to_camera(self._pointcloud_world, cam_idx, H, W)
        if u.numel() == 0:
            return depth
        depth_buf = torch.full((H, W), float("inf"), dtype=torch.float32)
        depth_buf[v, u] = torch.minimum(depth_buf[v, u], z.float())
        depth_buf[torch.isinf(depth_buf)] = 0.0
        return depth_buf

    def _hole_fill_depth(self, depth: torch.Tensor) -> torch.Tensor:
        """Fill holes with moving average guided by z-score threshold rule."""
        # depth: HxW CPU float
        k = self.config.depth_hole_fill_kappa
        M = self.config.depth_hole_fill_window
        if depth.numel() == 0:
            return depth
        dep = depth.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
        mask = (dep > 0).float()
        kernel = torch.ones((1, 1, M, M), dtype=dep.dtype)
        eps = 1e-8
        sum_ = F.conv2d(dep, kernel, padding=M // 2)
        count = F.conv2d(mask, kernel, padding=M // 2)
        mu = sum_ / torch.clamp(count, min=eps)
        sumsq = F.conv2d(dep * dep, kernel, padding=M // 2)
        var = sumsq / torch.clamp(count, min=eps) - mu * mu
        var = torch.clamp(var, min=0.0)
        sigma = torch.sqrt(var + 1e-8)

        # For holes (mask==0), decide fill if mu/sigma > k
        fill_mask = (mask == 0) & ((mu / torch.clamp(sigma, min=1e-6)) > k)
        filled = dep.clone()
        filled[fill_mask] = mu[fill_mask]

        return filled.squeeze(0).squeeze(0)

    def _estimate_depth_for_eval_camera(self, cam_idx: int) -> Optional[torch.Tensor]:
        if cam_idx in self._depth_cache:
            return self._depth_cache[cam_idx]
        depth = self._depth_from_pointcloud_for_camera(cam_idx)
        if depth is None:
            return None
        depth_filled = self._hole_fill_depth(depth)
        self._depth_cache[cam_idx] = depth_filled
        return depth_filled

    def _build_offline_pointcloud(self) -> None:
        """Build and refine point cloud from a subset of training views with GT depth."""
        if self.train_dataset is None:
            return
        cameras = self.train_dataset.cameras
        metadata = self.train_dataset.metadata
        print(metadata)
        exit(0)
        num_images = len(self.train_dataset)
        if num_images == 0:
            return
        # Ensure train dataset provides depth
        # Expect per-image "depth_image" in dataset.get_data.
        # We'll access raw on-demand through dataloader later; here, we use dataset cache if available.
        # Fallback: skip if no depth info in dataset.

        try:
            # Probe first item for depth existence
            first = self.train_dataset[0]
            has_depth = "depth_image" in first
        except Exception:
            has_depth = False

        if not has_depth:
            self._pointcloud_world = None
            return

        # Choose uniformly spaced subset
        K = max(1, min(self.config.pc_offline_num_views, num_images))
        idxs = torch.linspace(0, num_images - 1, steps=K).long().tolist()

        pointcloud_list: List[torch.Tensor] = []
        stride = max(1, int(self.config.pc_stride))

        # Seed cloud from subset views
        for idx in idxs:
            data = self.train_dataset[idx]
            depth = data["depth_image"]  # HxW
            if isinstance(depth, torch.Tensor):
                dep = depth.detach().cpu().float()
            else:
                dep = torch.tensor(depth).float()
            pts = self._backproject_depth_to_points(dep, idx, stride=stride)
            if pts.numel() > 0:
                pointcloud_list.append(pts)

        if len(pointcloud_list) == 0:
            self._pointcloud_world = torch.empty((0, 3), dtype=torch.float32)
            return

        cloud = torch.cat(pointcloud_list, dim=0)  # (N,3)
        tau = float(self.config.pc_tau)

        # Refinement: iterate over adjacent subset views
        for i in range(len(idxs) - 1):
            cur_idx = idxs[i]
            nxt_idx = idxs[i + 1]
            H, W = self._get_camera_hw(cameras, nxt_idx)
            # Predicted depth from current cloud to next view
            pred_depth = self._depth_from_pointcloud_for_camera(nxt_idx)
            if pred_depth is None:
                continue
            # Compare with GT depth of next view
            nxt_data = self.train_dataset[nxt_idx]
            gt_depth = nxt_data["depth_image"]
            if not isinstance(gt_depth, torch.Tensor):
                gt_depth = torch.tensor(gt_depth).float()
            gt_depth = gt_depth.detach().cpu().float()
            if pred_depth.shape != gt_depth.shape:
                gt_depth = gt_depth
            diff = torch.zeros_like(gt_depth)
            valid_pred = pred_depth > 0
            diff[valid_pred] = torch.abs(pred_depth[valid_pred] - gt_depth[valid_pred])
            # Where condition violated OR no prediction, add backprojected points
            need_add = (~valid_pred) | (diff > tau)
            if need_add.any():
                # Subsample to limit growth
                ys, xs = torch.where(need_add)
                ys = ys[::stride]
                xs = xs[::stride]
                if ys.numel() > 0:
                    z = gt_depth[ys, xs]
                    # Build sparse depth image for backprojection
                    sparse_depth = torch.zeros_like(gt_depth)
                    sparse_depth[ys, xs] = z
                    pts_new = self._backproject_depth_to_points(sparse_depth, nxt_idx, stride=1)
                    if pts_new.numel() > 0:
                        cloud = torch.cat([cloud, pts_new], dim=0)

        self._pointcloud_world = cloud.detach().cpu()

    # =========================
    # Near-surface range on rays
    # =========================

    def _compute_ray_t_from_depth(
        self, ray_bundle: RayBundle, indices: torch.Tensor, depth_vals: torch.Tensor
    ) -> torch.Tensor:
        """Compute per-ray t along the ray to reach the surface point defined by per-pixel depth (z along camera).
        We backproject pixel depth to world P, then t = dot(P - o, d).
        Returns t_center shape (N,) on ray device."""
        assert self.train_dataset is not None
        device = ray_bundle.origins.device
        N = indices.shape[0]
        cams = self.train_dataset.cameras
        t_list = []
        for i in range(N):
            im_idx = int(indices[i, 0].item())
            y = float(indices[i, 1].item())
            x = float(indices[i, 2].item())
            z = float(depth_vals[i].item())
            if z <= 0.0:
                t_list.append(torch.tensor(0.0, device=device))
                continue
            fx, fy, cx, cy = self._get_intrinsics(cams, im_idx)
            Tcw = self._get_Tcw(cams, im_idx).to(device)
            # Camera point
            x_cam = (x - cx) / fx * z
            y_cam = (y - cy) / fy * z
            P_cam = torch.tensor([x_cam, y_cam, z, 1.0], device=device, dtype=ray_bundle.origins.dtype)
            P_world = (Tcw @ P_cam)[0:3]
            o = ray_bundle.origins[i]
            d = ray_bundle.directions[i]
            t_val = torch.dot((P_world - o), d / (d.norm() + 1e-8))  # project onto dir
            t_list.append(t_val)
        t_center = torch.stack(t_list, dim=0)
        # Clamp to positive
        t_center = torch.clamp(t_center, min=1e-6)
        return t_center

    def _inject_near_surface_range(
        self, ray_bundle: RayBundle, indices: torch.Tensor, depths: torch.Tensor
    ) -> RayBundle:
        """Modify ray_bundle.nears/fars based on per-ray depth with jitter gamma in [0, 2a/N]"""
        if not self.config.enable_near_surface_sampling:
            return ray_bundle
        device = ray_bundle.origins.device
        t_center = self._compute_ray_t_from_depth(ray_bundle, indices, depths)
        alpha = float(self.config.near_surface_alpha)
        N = max(1, int(self.config.ns_num_samples))
        jitter_amp = (2.0 * alpha) / float(N)
        gamma = torch.zeros_like(t_center)
        if self.config.jitter_near_surface:
            gamma = torch.rand_like(t_center) * jitter_amp
        nears = torch.clamp(t_center - alpha + gamma, min=1e-6)
        fars = torch.clamp(t_center + alpha + gamma, min=nears + 1e-6)
        ray_bundle.nears = nears
        ray_bundle.fars = fars
        return ray_bundle

    # =========================
    # Dataloaders
    # =========================

    def next_train(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the train dataloader."""
        self.train_count += 1
        image_batch = next(self.iter_train_image_dataloader)
        assert self.train_pixel_sampler is not None
        assert isinstance(image_batch, dict)
        batch = self.train_pixel_sampler.sample(image_batch)
        ray_indices = batch["indices"]  # [N, 3] -> (image_index, y, x)
        ray_bundle = self.train_ray_generator(ray_indices)

        # Inject near/far based on GT depth if available
        depth_key = "depth_image" if "depth_image" in batch else ("depth" if "depth" in batch else None)
        if depth_key is not None:
            depths = batch[depth_key].reshape(-1).to(ray_bundle.origins.device).float()
            ray_bundle = self._inject_near_surface_range(ray_bundle, ray_indices, depths)

        return ray_bundle, batch

    def next_eval(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch from the eval dataloader with near-surface range estimated from point cloud."""
        self.eval_count += 1
        image_batch = next(self.iter_eval_image_dataloader)
        assert self.eval_pixel_sampler is not None
        assert isinstance(image_batch, dict)
        batch = self.eval_pixel_sampler.sample(image_batch)
        ray_indices = batch["indices"]
        ray_bundle = self.eval_ray_generator(ray_indices)

        # Prefer GT depth if present (e.g., validation with GT depth)
        depth_key = "depth_image" if "depth_image" in batch else ("depth" if "depth" in batch else None)
        if depth_key is not None:
            depths = batch[depth_key].reshape(-1).to(ray_bundle.origins.device).float()
            ray_bundle = self._inject_near_surface_range(ray_bundle, ray_indices, depths)
            return ray_bundle, batch

        # Else, estimate from point cloud if available
        if self.config.use_pointcloud_for_eval and self._pointcloud_world is not None:
            # For each ray, fetch estimated depth from its camera's map at (y,x)
            # ray_indices: [N,3] (image_index, y, x)
            cam_indices = ray_indices[:, 0].tolist()
            ys = ray_indices[:, 1].long()
            xs = ray_indices[:, 2].long()
            depths_list = []
            for i, cam_idx in enumerate(cam_indices):
                dep_map = self._estimate_depth_for_eval_camera(int(cam_idx))
                if dep_map is None:
                    depths_list.append(torch.tensor(0.0))
                else:
                    H, W = dep_map.shape
                    y = int(max(0, min(H - 1, int(ys[i].item()))))
                    x = int(max(0, min(W - 1, int(xs[i].item()))))
                    depths_list.append(dep_map[y, x])
            depths = torch.stack([d.clone().detach() if isinstance(d, torch.Tensor) else torch.tensor(d) for d in depths_list], dim=0)
            depths = depths.to(ray_bundle.origins.device).float()
            ray_bundle = self._inject_near_surface_range(ray_bundle, ray_indices, depths)

        return ray_bundle, batch
