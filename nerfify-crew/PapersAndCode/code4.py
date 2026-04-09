"kplanes.py"

"""
Implementation of K-Planes (https://sarafridov.github.io/K-Planes/).
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Type

import numpy as np
import torch
from torch.nn import Parameter
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from typing_extensions import Literal

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.configs.config_utils import to_immutable_dict
from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.model_components.losses import MSELoss, distortion_loss, interlevel_loss
from nerfstudio.model_components.ray_samplers import (
    ProposalNetworkSampler,
    UniformLinDispPiecewiseSampler,
    UniformSampler,
)
from nerfstudio.model_components.renderers import (
    AccumulationRenderer,
    DepthRenderer,
    RGBRenderer,
)
from nerfstudio.model_components.scene_colliders import NearFarCollider
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import colormaps, misc

from .kplanes_field import KPlanesDensityField, KPlanesField


@dataclass
class KPlanesModelConfig(ModelConfig):
    """K-Planes Model Config"""

    _target: Type = field(default_factory=lambda: KPlanesModel)

    near_plane: float = 2.0
    """How far along the ray to start sampling."""

    far_plane: float = 6.0
    """How far along the ray to stop sampling."""

    grid_base_resolution: List[int] = field(default_factory=lambda: [128, 128, 128])
    """Base grid resolution."""

    grid_feature_dim: int = 32
    """Dimension of feature vectors stored in grid."""

    multiscale_res: List[int] = field(default_factory=lambda: [1, 2, 4])
    """Multiscale grid resolutions."""

    is_contracted: bool = False
    """Whether to use scene contraction (set to true for unbounded scenes)."""

    concat_features_across_scales: bool = True
    """Whether to concatenate features at different scales."""

    linear_decoder: bool = False
    """Whether to use a linear decoder instead of an MLP."""

    linear_decoder_layers: Optional[int] = 1
    """Number of layers in linear decoder"""

    # proposal sampling arguments
    num_proposal_iterations: int = 2
    """Number of proposal network iterations."""

    use_same_proposal_network: bool = False
    """Use the same proposal network. Otherwise use different ones."""

    proposal_net_args_list: List[Dict] = field(
        default_factory=lambda: [
            {"num_output_coords": 8, "resolution": [64, 64, 64]},
            {"num_output_coords": 8, "resolution": [128, 128, 128]},
        ]
    )
    """Arguments for the proposal density fields."""

    num_proposal_samples: Optional[Tuple[int]] = (256, 128)
    """Number of samples per ray for each proposal network."""

    num_samples: Optional[int] = 48
    """Number of samples per ray used for rendering."""

    single_jitter: bool = False
    """Whether use single jitter or not for the proposal networks."""

    proposal_warmup: int = 5000
    """Scales n from 1 to proposal_update_every over this many steps."""

    proposal_update_every: int = 5
    """Sample every n steps after the warmup."""

    use_proposal_weight_anneal: bool = True
    """Whether to use proposal weight annealing."""

    proposal_weights_anneal_slope: float = 10.0
    """Slope of the annealing function for the proposal weights."""

    proposal_weights_anneal_max_num_iters: int = 1000
    """Max num iterations for the annealing function."""

    appearance_embedding_dim: int = 0
    """Dimension of appearance embedding. Set to 0 to disable."""

    use_average_appearance_embedding: bool = True
    """Whether to use average appearance embedding or zeros for inference."""

    background_color: Literal["random", "last_sample", "black", "white"] = "white"
    """The background color as RGB."""

    loss_coefficients: Dict[str, float] = to_immutable_dict(
        {
            "img": 1.0,
            "interlevel": 1.0,
            "distortion": 0.001,
            "plane_tv": 0.0001,
            "plane_tv_proposal_net": 0.0001,
            "l1_time_planes": 0.0001,
            "l1_time_planes_proposal_net": 0.0001,
            "time_smoothness": 0.1,
            "time_smoothness_proposal_net": 0.001,
        }
    )
    """Loss coefficients."""


class KPlanesModel(Model):
    config: KPlanesModelConfig
    """K-Planes model

    Args:
        config: K-Planes configuration to instantiate model
    """

    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()

        if self.config.is_contracted:
            scene_contraction = SceneContraction(order=float("inf"))
        else:
            scene_contraction = None

        # Fields
        self.field = KPlanesField(
            self.scene_box.aabb,
            num_images=self.num_train_data,
            grid_base_resolution=self.config.grid_base_resolution,
            grid_feature_dim=self.config.grid_feature_dim,
            concat_across_scales=self.config.concat_features_across_scales,
            multiscale_res=self.config.multiscale_res,
            spatial_distortion=scene_contraction,
            appearance_embedding_dim=self.config.appearance_embedding_dim,
            use_average_appearance_embedding=self.config.use_average_appearance_embedding,
            linear_decoder=self.config.linear_decoder,
            linear_decoder_layers=self.config.linear_decoder_layers,
        )

        self.density_fns = []
        num_prop_nets = self.config.num_proposal_iterations
        # Build the proposal network(s)
        self.proposal_networks = torch.nn.ModuleList()
        if self.config.use_same_proposal_network:
            assert len(self.config.proposal_net_args_list) == 1, "Only one proposal network is allowed."
            prop_net_args = self.config.proposal_net_args_list[0]
            network = KPlanesDensityField(
                self.scene_box.aabb,
                spatial_distortion=scene_contraction,
                linear_decoder=self.config.linear_decoder,
                **prop_net_args,
            )
            self.proposal_networks.append(network)
            self.density_fns.extend([network.density_fn for _ in range(num_prop_nets)])
        else:
            for i in range(num_prop_nets):
                prop_net_args = self.config.proposal_net_args_list[min(i, len(self.config.proposal_net_args_list) - 1)]
                network = KPlanesDensityField(
                    self.scene_box.aabb,
                    spatial_distortion=scene_contraction,
                    linear_decoder=self.config.linear_decoder,
                    **prop_net_args,
                )
                self.proposal_networks.append(network)
            self.density_fns.extend([network.density_fn for network in self.proposal_networks])

        # Samplers
        def update_schedule(step):
            return np.clip(
                np.interp(step, [0, self.config.proposal_warmup], [0, self.config.proposal_update_every]),
                1,
                self.config.proposal_update_every,
            )

        if self.config.is_contracted:
            initial_sampler = UniformLinDispPiecewiseSampler(single_jitter=self.config.single_jitter)
        else:
            initial_sampler = UniformSampler(single_jitter=self.config.single_jitter)

        self.proposal_sampler = ProposalNetworkSampler(
            num_nerf_samples_per_ray=self.config.num_samples,
            num_proposal_samples_per_ray=self.config.num_proposal_samples,
            num_proposal_network_iterations=self.config.num_proposal_iterations,
            single_jitter=self.config.single_jitter,
            update_sched=update_schedule,
            initial_sampler=initial_sampler,
        )

        # Collider
        self.collider = NearFarCollider(near_plane=self.config.near_plane, far_plane=self.config.far_plane)

        # renderers
        self.renderer_rgb = RGBRenderer(background_color=self.config.background_color)
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer()

        # losses
        self.rgb_loss = MSELoss()

        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)
        self.temporal_distortion = len(self.config.grid_base_resolution) == 4  # for viewer

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {
            "proposal_networks": list(self.proposal_networks.parameters()),
            "fields": list(self.field.parameters())
        }
        return param_groups

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        callbacks = []
        if self.config.use_proposal_weight_anneal:
            # anneal the weights of the proposal network before doing PDF sampling
            N = self.config.proposal_weights_anneal_max_num_iters

            def set_anneal(step):
                # https://arxiv.org/pdf/2111.12077.pdf eq. 18
                train_frac = np.clip(step / N, 0, 1)
                bias = lambda x, b: (b * x) / ((b - 1) * x + 1)
                anneal = bias(train_frac, self.config.proposal_weights_anneal_slope)
                self.proposal_sampler.set_anneal(anneal)

            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=set_anneal,
                )
            )
            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=self.proposal_sampler.step_cb,
                )
            )
        return callbacks

    def get_outputs(self, ray_bundle: RayBundle):
        density_fns = self.density_fns
        if ray_bundle.times is not None:
            density_fns = [functools.partial(f, times=ray_bundle.times) for f in density_fns]
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler(
            ray_bundle, density_fns=density_fns
        )
        field_outputs = self.field(ray_samples)

        weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])
        weights_list.append(weights)
        ray_samples_list.append(ray_samples)

        rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)
        depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
        accumulation = self.renderer_accumulation(weights=weights)

        outputs = {
            "rgb": rgb,
            "accumulation": accumulation,
            "depth": depth,
        }

        # These use a lot of GPU memory, so we avoid storing them for eval.
        if self.training:
            outputs["weights_list"] = weights_list
            outputs["ray_samples_list"] = ray_samples_list

        for i in range(self.config.num_proposal_iterations):
            outputs[f"prop_depth_{i}"] = self.renderer_depth(
                weights=weights_list[i], ray_samples=ray_samples_list[i]
            )

        return outputs

    def get_metrics_dict(self, outputs, batch):
        image = batch["image"].to(self.device)
        image = self.renderer_rgb.blend_background(image)  # Blend if RGBA

        metrics_dict = {
            "psnr": self.psnr(outputs["rgb"], image)
        }
        if self.training:
            metrics_dict["interlevel"] = interlevel_loss(outputs["weights_list"], outputs["ray_samples_list"])
            metrics_dict["distortion"] = distortion_loss(outputs["weights_list"], outputs["ray_samples_list"])

            prop_grids = [p.grids.plane_coefs for p in self.proposal_networks]
            field_grids = [g.plane_coefs for g in self.field.grids]

            metrics_dict["plane_tv"] = space_tv_loss(field_grids)
            metrics_dict["plane_tv_proposal_net"] = space_tv_loss(prop_grids)

            if len(self.config.grid_base_resolution) == 4:
                metrics_dict["l1_time_planes"] = l1_time_planes(field_grids)
                metrics_dict["l1_time_planes_proposal_net"] = l1_time_planes(prop_grids)
                metrics_dict["time_smoothness"] = time_smoothness(field_grids)
                metrics_dict["time_smoothness_proposal_net"] = time_smoothness(prop_grids)

        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        image = batch["image"].to(self.device)
        pred_rgb, image = self.renderer_rgb.blend_background_for_loss_computation(
            pred_image=outputs["rgb"],
            pred_accumulation=outputs["accumulation"],
            gt_image=image,
        )

        loss_dict = {"rgb": self.rgb_loss(image, pred_rgb)}
        if self.training:
            for key in self.config.loss_coefficients:
                if key in metrics_dict:
                    loss_dict[key] = metrics_dict[key].clone()

            loss_dict = misc.scale_dict(loss_dict, self.config.loss_coefficients)

        return loss_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        image = batch["image"].to(self.device)
        image = self.renderer_rgb.blend_background(image)

        rgb = outputs["rgb"]
        acc = colormaps.apply_colormap(outputs["accumulation"])
        depth = colormaps.apply_depth_colormap(outputs["depth"], accumulation=outputs["accumulation"])

        combined_rgb = torch.cat([image, rgb], dim=1)
        combined_acc = torch.cat([acc], dim=1)
        combined_depth = torch.cat([depth], dim=1)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        image = torch.moveaxis(image, -1, 0)[None, ...]
        rgb = torch.moveaxis(rgb, -1, 0)[None, ...]

        # all of these metrics will be logged as scalars
        metrics_dict = {
            "psnr": float(self.psnr(image, rgb).item()),
            "ssim": float(self.ssim(image, rgb)),
            "lpips": float(self.lpips(image, rgb))
        }
        images_dict = {"img": combined_rgb, "accumulation": combined_acc, "depth": combined_depth}

        for i in range(self.config.num_proposal_iterations):
            key = f"prop_depth_{i}"
            prop_depth_i = colormaps.apply_depth_colormap(
                outputs[key],
                accumulation=outputs["accumulation"],
            )
            images_dict[key] = prop_depth_i

        return metrics_dict, images_dict


def compute_plane_tv(t: torch.Tensor, only_w: bool = False) -> float:
    """Computes total variance across a plane.

    Args:
        t: Plane tensor
        only_w: Whether to only compute total variance across w dimension

    Returns:
        Total variance
    """
    _, h, w = t.shape
    w_tv = torch.square(t[..., :, 1:] - t[..., :, : w - 1]).mean()

    if only_w:
        return w_tv

    h_tv = torch.square(t[..., 1:, :] - t[..., : h - 1, :]).mean()
    return h_tv + w_tv


def space_tv_loss(multi_res_grids: List[torch.Tensor]) -> float:
    """Computes total variance across each spatial plane in the grids.

    Args:
        multi_res_grids: Grids to compute total variance over

    Returns:
        Total variance
    """

    total = 0.0
    num_planes = 0
    for grids in multi_res_grids:
        if len(grids) == 3:
            spatial_planes = {0, 1, 2}
        else:
            spatial_planes = {0, 1, 3}

        for grid_id, grid in enumerate(grids):
            if grid_id in spatial_planes:
                total += compute_plane_tv(grid)
            else:
                # Space is the last dimension for space-time planes.
                total += compute_plane_tv(grid, only_w=True)
            num_planes += 1
    return total / num_planes


def l1_time_planes(multi_res_grids: List[torch.Tensor]) -> float:
    """Computes the L1 distance from the multiplicative identity (1) for spatiotemporal planes.

    Args:
        multi_res_grids: Grids to compute L1 distance over

    Returns:
         L1 distance from the multiplicative identity (1)
    """
    time_planes = [2, 4, 5]  # These are the spatiotemporal planes
    total = 0.0
    num_planes = 0
    for grids in multi_res_grids:
        for grid_id in time_planes:
            total += torch.abs(1 - grids[grid_id]).mean()
            num_planes += 1

    return total / num_planes


def compute_plane_smoothness(t: torch.Tensor) -> float:
    """Computes smoothness across the temporal axis of a plane

    Args:
        t: Plane tensor

    Returns:
        Time smoothness
    """
    _, h, _ = t.shape
    # Convolve with a second derivative filter, in the time dimension which is dimension 2
    first_difference = t[..., 1:, :] - t[..., : h - 1, :]  # [c, h-1, w]
    second_difference = first_difference[..., 1:, :] - first_difference[..., : h - 2, :]  # [c, h-2, w]
    # Take the L2 norm of the result
    return torch.square(second_difference).mean()


def time_smoothness(multi_res_grids: List[torch.Tensor]) -> float:
    """Computes smoothness across each time plane in the grids.

    Args:
        multi_res_grids: Grids to compute time smoothness over

    Returns:
        Time smoothness
    """
    total = 0.0
    num_planes = 0
    for grids in multi_res_grids:
        time_planes = [2, 4, 5]  # These are the spatiotemporal planes
        for grid_id in time_planes:
            total += compute_plane_smoothness(grids[grid_id])
            num_planes += 1

    return total / num_planes

"kplanes_configs.py"
from __future__ import annotations

from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManagerConfig
from nerfstudio.data.dataparsers.blender_dataparser import BlenderDataParserConfig
from nerfstudio.data.dataparsers.dnerf_dataparser import DNeRFDataParserConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import CosineDecaySchedulerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from nerfstudio.plugins.types import MethodSpecification

from .kplanes import KPlanesModelConfig


kplanes_method = MethodSpecification(
    config=TrainerConfig(
        method_name="kplanes",
        steps_per_eval_batch=500,
        steps_per_save=2000,
        steps_per_eval_all_images=30000,
        max_num_iterations=30001,
        mixed_precision=True,
        pipeline=VanillaPipelineConfig(
            datamanager=VanillaDataManagerConfig(
                dataparser=BlenderDataParserConfig(),
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=4096,
            ),
            model=KPlanesModelConfig(
                eval_num_rays_per_chunk=1 << 15,
                grid_base_resolution=[128, 128, 128],
                grid_feature_dim=32,
                multiscale_res=[1, 2, 4],
                proposal_net_args_list=[
                    {"num_output_coords": 8, "resolution": [128, 128, 128]},
                    {"num_output_coords": 8, "resolution": [256, 256, 256]}
                ],
                loss_coefficients={
                    "interlevel": 1.0,
                    "distortion": 0.01,
                    "plane_tv": 0.01,
                    "plane_tv_proposal_net": 0.0001,
                },
                background_color="white",
            ),
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-12),
                "scheduler": CosineDecaySchedulerConfig(warm_up_end=512, max_steps=30000),
            },
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-12),
                "scheduler": CosineDecaySchedulerConfig(warm_up_end=512, max_steps=30000),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="K-Planes NeRF model for static scenes"
)

kplanes_dynamic_method = MethodSpecification(
    config=TrainerConfig(
        method_name="kplanes-dynamic",
        steps_per_eval_batch=500,
        steps_per_save=2000,
        steps_per_eval_all_images=30000,
        max_num_iterations=30001,
        mixed_precision=True,
        pipeline=VanillaPipelineConfig(
            datamanager=VanillaDataManagerConfig(
                dataparser=DNeRFDataParserConfig(),
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=4096,
                camera_res_scale_factor=0.5,  # DNeRF train on 400x400
            ),
            model=KPlanesModelConfig(
                eval_num_rays_per_chunk=1 << 15,
                grid_base_resolution=[128, 128, 128, 25],  # time-resolution should be half the time-steps
                grid_feature_dim=32,
                multiscale_res=[1, 2, 4],
                proposal_net_args_list=[
                    # time-resolution should be half the time-steps
                    {"num_output_coords": 8, "resolution": [128, 128, 128, 25]},
                    {"num_output_coords": 8, "resolution": [256, 256, 256, 25]},
                ],
                loss_coefficients={
                    "interlevel": 1.0,
                    "distortion": 0.01,
                    "plane_tv": 0.1,
                    "plane_tv_proposal_net": 0.0001,
                    "l1_time_planes": 0.001,
                    "l1_time_planes_proposal_net": 0.0001,
                    "time_smoothness": 0.1,
                    "time_smoothness_proposal_net": 0.001,
                },
            ),
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-12),
                "scheduler": CosineDecaySchedulerConfig(warm_up_end=512, max_steps=30000),
            },
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-12),
                "scheduler": CosineDecaySchedulerConfig(warm_up_end=512, max_steps=30000),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="K-Planes NeRF model for dynamic scenes"
)

"kplanes_field.py"

"""
Fields for K-Planes (https://sarafridov.github.io/K-Planes/).
"""

from typing import Dict, Iterable, List, Optional, Tuple, Sequence

import torch
from rich.console import Console
from torch import nn
from torchtyping import TensorType

from nerfstudio.cameras.rays import RaySamples, Frustums
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.field_components.activations import trunc_exp
from nerfstudio.field_components.embedding import Embedding
from nerfstudio.field_components.encodings import KPlanesEncoding
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SpatialDistortion
from nerfstudio.fields.base_field import Field, get_normalized_directions

try:
    import tinycudann as tcnn
except ImportError:
    # tinycudann module doesn't exist
    pass

CONSOLE = Console(width=120)


def interpolate_ms_features(
    pts: torch.Tensor,
    grid_encodings: Iterable[KPlanesEncoding],
    concat_features: bool,
) -> torch.Tensor:
    """Combines/interpolates features across multiple dimensions and scales.

    Args:
        pts: Coordinates to query
        grid_encodings: Grid encodings to query
        concat_features: Whether to concatenate features at different scales

    Returns:
        Feature vectors
    """

    multi_scale_interp = [] if concat_features else 0.0
    for grid in grid_encodings:
        grid_features = grid(pts)

        if concat_features:
            multi_scale_interp.append(grid_features)
        else:
            multi_scale_interp = multi_scale_interp + grid_features

    if concat_features:
        multi_scale_interp = torch.cat(multi_scale_interp, dim=-1)

    return multi_scale_interp


class KPlanesField(Field):
    """K-Planes field.

    Args:
        aabb: Parameters of scene aabb bounds
        num_images: How many images exist in the dataset
        geo_feat_dim: Dimension of 'geometry' features. Controls output dimension of sigma network
        grid_base_resolution: Base grid resolution
        grid_feature_dim: Dimension of feature vectors stored in grid
        concat_across_scales: Whether to concatenate features at different scales
        multiscale_res: Multiscale grid resolutions
        spatial_distortion: Spatial distortion to apply to the scene
        appearance_embedding_dim: Dimension of appearance embedding. Set to 0 to disable
        use_average_appearance_embedding: Whether to use average appearance embedding or zeros for inference
        linear_decoder: Whether to use a linear decoder instead of an MLP
        linear_decoder_layers: Number of layers in linear decoder
    """

    def __init__(
        self,
        aabb: TensorType,
        num_images: int,
        geo_feat_dim: int = 15,  # TODO: This should be removed
        concat_across_scales: bool = True,  # TODO: Maybe this should be removed
        grid_base_resolution: Sequence[int] = (128, 128, 128),
        grid_feature_dim: int = 32,
        multiscale_res: Sequence[int] = (1, 2, 4),
        spatial_distortion: Optional[SpatialDistortion] = None,
        appearance_embedding_dim: int = 0,
        use_average_appearance_embedding: bool = True,
        linear_decoder: bool = False,
        linear_decoder_layers: Optional[int] = None,
    ) -> None:

        super().__init__()

        self.register_buffer("aabb", aabb)
        self.num_images = num_images
        self.geo_feat_dim = geo_feat_dim
        self.grid_base_resolution = list(grid_base_resolution)
        self.concat_across_scales = concat_across_scales
        self.spatial_distortion = spatial_distortion
        self.linear_decoder = linear_decoder
        self.has_time_planes = len(grid_base_resolution) > 3

        # Init planes
        self.grids = nn.ModuleList()
        for res in multiscale_res:
            # Resolution fix: multi-res only on spatial planes
            resolution = [r * res for r in self.grid_base_resolution[:3]] + self.grid_base_resolution[3:]
            self.grids.append(KPlanesEncoding(resolution, grid_feature_dim))
        self.feature_dim = (
            grid_feature_dim * len(multiscale_res) if self.concat_across_scales
            else grid_feature_dim
        )

        # Init appearance code-related parameters
        self.appearance_embedding_dim = appearance_embedding_dim
        if self.appearance_embedding_dim > 0:
            assert self.num_images is not None, "'num_images' must not be None when using appearance embedding"
            self.appearance_ambedding = Embedding(self.num_images, self.appearance_embedding_dim)
            self.use_average_appearance_embedding = use_average_appearance_embedding  # for test-time

        # Init decoder network
        if self.linear_decoder:
            assert linear_decoder_layers is not None
            # The NN learns a basis that is used instead of spherical harmonics
            # Input is an encoded view direction, output is weights for combining the color
            # features into RGB.
            # Architecture based on instant-NGP
            self.color_basis = tcnn.Network(
                n_input_dims=3 + self.appearance_embedding_dim,
                n_output_dims=3 * self.feature_dim,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": 128,
                    "n_hidden_layers": linear_decoder_layers,
                },
            )
            # sigma_net just does a linear transformation on the features to get density
            self.sigma_net = tcnn.Network(
                n_input_dims=self.feature_dim,
                n_output_dims=1,
                network_config={
                    "otype": "CutlassMLP",
                    "activation": "None",
                    "output_activation": "None",
                    "n_neurons": 128,
                    "n_hidden_layers": 0,
                },
            )
        else:
            self.sigma_net = tcnn.Network(
                n_input_dims=self.feature_dim,
                n_output_dims=self.geo_feat_dim + 1,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": 64,
                    "n_hidden_layers": 1,
                },
            )
            self.direction_encoding = tcnn.Encoding(
                n_input_dims=3,
                encoding_config={
                    "otype": "SphericalHarmonics",
                    "degree": 4,
                },
            )
            in_dim_color = (
                self.direction_encoding.n_output_dims + self.geo_feat_dim + self.appearance_embedding_dim
            )
            self.color_net = tcnn.Network(
                n_input_dims=in_dim_color,
                n_output_dims=3,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "Sigmoid",
                    "n_neurons": 64,
                    "n_hidden_layers": 2,
                },
            )

    def get_density(self, ray_samples: RaySamples) -> Tuple[TensorType, TensorType]:
        """Computes and returns the densities."""
        positions = ray_samples.frustums.get_positions()
        if self.spatial_distortion is not None:
            positions = self.spatial_distortion(positions)
            positions = positions / 2  # from [-2, 2] to [-1, 1]
        else:
            # From [0, 1] to [-1, 1]
            positions = SceneBox.get_normalized_positions(positions, self.aabb) * 2.0 - 1.0

        if self.has_time_planes:
            assert ray_samples.times is not None, "Initialized model with time-planes, but no time data is given"
            # Normalize timestamps from [0, 1] to [-1, 1]
            timestamps = ray_samples.times * 2.0 - 1.0
            positions = torch.cat((positions, timestamps), dim=-1)  # [n_rays, n_samples, 4]

        positions_flat = positions.view(-1, positions.shape[-1])
        features = interpolate_ms_features(
            positions_flat, grid_encodings=self.grids, concat_features=self.concat_across_scales
        )
        if len(features) < 1:
            features = torch.zeros((0, 1), device=features.device, requires_grad=True)
        if self.linear_decoder:
            density_before_activation = self.sigma_net(features).view(*ray_samples.frustums.shape, -1)
        else:
            features = self.sigma_net(features).view(*ray_samples.frustums.shape, -1)
            features, density_before_activation = torch.split(features, [self.geo_feat_dim, 1], dim=-1)

        # Rectifying the density with an exponential is much more stable than a ReLU or
        # softplus, because it enables high post-activation (float32) density outputs
        # from smaller internal (float16) parameters.
        density = trunc_exp(density_before_activation.to(positions) - 1)
        return density, features

    def get_outputs(
        self, ray_samples: RaySamples, density_embedding: Optional[TensorType] = None
    ) -> Dict[FieldHeadNames, TensorType]:
        assert density_embedding is not None

        output_shape = ray_samples.frustums.shape
        directions = ray_samples.frustums.directions.reshape(-1, 3)

        if self.linear_decoder:
            color_features = [density_embedding]
        else:
            directions = get_normalized_directions(directions)
            d = self.direction_encoding(directions)
            color_features = [d, density_embedding.view(-1, self.geo_feat_dim)]

        if self.appearance_embedding_dim > 0:
            if self.training:
                assert ray_samples.camera_indices is not None
                camera_indices = ray_samples.camera_indices.squeeze()
                embedded_appearance = self.appearance_ambedding(camera_indices)
            elif self.use_average_appearance_embedding:
                embedded_appearance = torch.ones(
                    (*output_shape, self.appearance_embedding_dim),
                    device=directions.device,
                ) * self.appearance_ambedding.mean(dim=0)
            else:
                embedded_appearance = torch.zeros(
                    (*output_shape, self.appearance_embedding_dim),
                    device=directions.device,
                )

            if not self.linear_decoder:
                color_features.append(embedded_appearance)

        color_features = torch.cat(color_features, dim=-1)
        if self.linear_decoder:
            basis_input = directions
            if self.appearance_ambedding_dim > 0:
                basis_input = torch.cat([directions, embedded_appearance], dim=-1)
            basis_values = self.color_basis(basis_input) # [batch, color_feature_len * 3]
            basis_values = basis_values.view(basis_input.shape[0], 3, -1)  # [batch, color_feature_len, 3]
            rgb = torch.sum(color_features[:, None, :] * basis_values, dim=-1)  # [batch, 3]
            rgb = torch.sigmoid(rgb).view(*output_shape, -1).to(directions)
        else:
            rgb = self.color_net(color_features).view(*output_shape, -1)

        return {FieldHeadNames.RGB: rgb}


class KPlanesDensityField(Field):
    """A lightweight density field module.

    Args:
        aabb: Parameters of scene aabb bounds
        resolution: Grid resolution
        num_output_coords: dimension of grid feature vectors
        spatial_distortion: Spatial distortion to apply to the scene
        linear_decoder: Whether to use a linear decoder instead of an MLP
    """

    def __init__(
        self,
        aabb: TensorType,
        resolution: List[int],
        num_output_coords: int,
        spatial_distortion: Optional[SpatialDistortion] = None,
        linear_decoder: bool = False,
    ):
        super().__init__()

        self.register_buffer("aabb", aabb)

        self.spatial_distortion = spatial_distortion
        self.has_time_planes = len(resolution) > 3
        self.feature_dim = num_output_coords
        self.linear_decoder = linear_decoder

        self.grids = KPlanesEncoding(resolution, num_output_coords, init_a=0.1, init_b=0.15)

        self.sigma_net = tcnn.Network(
            n_input_dims=self.feature_dim,
            n_output_dims=1,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "None" if self.linear_decoder else "ReLU",
                "output_activation": "None",
                "n_neurons": 64,
                "n_hidden_layers": 1,
            },
        )

        CONSOLE.log(f"Initialized KPlaneDensityField. with time-planes={self.has_time_planes} - resolution={resolution}")

    # pylint: disable=arguments-differ
    def density_fn(self, positions: TensorType["bs":..., 3], times: Optional[TensorType["bs", 1]] = None) -> TensorType["bs":..., 1]:
        """Returns only the density. Overrides base function to add times in samples

        Args:
            positions: the origin of the samples/frustums
            times: the time of rays
        """
        if times is not None and (len(positions.shape) == 3 and len(times.shape) == 2):
            # position is [ray, sample, 3]; times is [ray, 1]
            times = times[:, None]  # RaySamples can handle the shape
        # Need to figure out a better way to descibe positions with a ray.
        ray_samples = RaySamples(
            frustums=Frustums(
                origins=positions,
                directions=torch.ones_like(positions),
                starts=torch.zeros_like(positions[..., :1]),
                ends=torch.zeros_like(positions[..., :1]),
                pixel_area=torch.ones_like(positions[..., :1]),
            ),
            times=times,
        )
        density, _ = self.get_density(ray_samples)
        return density

    def get_density(self, ray_samples: RaySamples) -> Tuple[TensorType, None]:
        """Computes and returns the densities."""
        positions = ray_samples.frustums.get_positions()
        if self.spatial_distortion is not None:
            positions = self.spatial_distortion(positions)
            positions = positions / 2  # from [-2, 2] to [-1, 1]
        else:
            # From [0, 1] to [-1, 1]
            positions = SceneBox.get_normalized_positions(positions, self.aabb) * 2.0 - 1.0

        if self.has_time_planes:
            assert ray_samples.times is not None, "Initialized model with time-planes, but no time data is given"
            # Normalize timestamps from [0, 1] to [-1, 1]
            timestamps = ray_samples.times * 2.0 - 1.0
            positions = torch.cat((positions, timestamps), dim=-1)  # [n_rays, n_samples, 4]

        positions_flat = positions.view(-1, positions.shape[-1])
        features = interpolate_ms_features(
            positions_flat, grid_encodings=[self.grids], concat_features=False
        )
        if len(features) < 1:
            features = torch.zeros((0, 1), device=features.device, requires_grad=True)
        density_before_activation = self.sigma_net(features).view(*ray_samples.frustums.shape, -1)
        density = trunc_exp(density_before_activation.to(positions) - 1)
        return density, None

    def get_outputs(self, ray_samples: RaySamples, density_embedding: Optional[TensorType] = None) -> dict:
        return {}