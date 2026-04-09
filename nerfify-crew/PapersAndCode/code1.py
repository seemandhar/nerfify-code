"""
nerfplayer_config.py
"""


from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.datamanagers.base_datamanager import (
    VanillaDataManager,
    VanillaDataManagerConfig,
)
from nerfstudio.data.dataparsers.dycheck_dataparser import DycheckDataParserConfig
from nerfstudio.data.datasets.depth_dataset import DepthDataset
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from nerfstudio.pipelines.dynamic_batch import DynamicBatchPipelineConfig
from nerfstudio.plugins.types import MethodSpecification

from nerfplayer.nerfplayer_nerfacto import NerfplayerNerfactoModelConfig
from nerfplayer.nerfplayer_ngp import NerfplayerNGPModelConfig


nerfplayer_nerfacto = MethodSpecification(
    config=TrainerConfig(
        method_name="nerfplayer-nerfacto",
        steps_per_eval_batch=500,
        steps_per_save=2000,
        max_num_iterations=30000,
        mixed_precision=True,
        pipeline=VanillaPipelineConfig(
            datamanager=VanillaDataManagerConfig(
                _target=VanillaDataManager[DepthDataset],
                dataparser=DycheckDataParserConfig(),
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=4096,
            ),
            model=NerfplayerNerfactoModelConfig(
                eval_num_rays_per_chunk=1 << 15,
                camera_optimizer=CameraOptimizerConfig(mode="SO3xR3"),
            ),
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
            },
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
            },
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=5000),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="NeRFPlayer with nerfacto backbone.",
)

nerfplayer_ngp = MethodSpecification(
    config=TrainerConfig(
        method_name="nerfplayer-ngp",
        steps_per_eval_batch=500,
        steps_per_save=2000,
        max_num_iterations=30000,
        mixed_precision=True,
        pipeline=DynamicBatchPipelineConfig(
            datamanager=VanillaDataManagerConfig(
                _target=VanillaDataManager[DepthDataset],
                dataparser=DycheckDataParserConfig(),
                train_num_rays_per_batch=8192,
            ),
            model=NerfplayerNGPModelConfig(
                eval_num_rays_per_chunk=8192,
                grid_levels=1,
                alpha_thre=0.0,
                render_step_size=0.001,
                disable_scene_contraction=True,
                near_plane=0.01,
            ),
        ),
        optimizers={
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
            }
        },
        viewer=ViewerConfig(num_rays_per_chunk=64000),
        vis="viewer",
    ),
    description="NeRFPlayer with InstantNGP backbone.",
)


"""
nerfplayer_nerfacto.py
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Sequence, Type, cast

import numpy as np
import torch
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.model_components.losses import (
    MSELoss,
    interlevel_loss,
    orientation_loss,
    pred_normal_loss,
)
from nerfstudio.model_components.ray_samplers import ProposalNetworkSampler
from nerfstudio.model_components.renderers import (
    AccumulationRenderer,
    DepthRenderer,
    NormalsRenderer,
    RGBRenderer,
)
from nerfstudio.model_components.scene_colliders import NearFarCollider
from nerfstudio.model_components.shaders import NormalsShader
from nerfstudio.models.base_model import Model
from nerfstudio.models.nerfacto import NerfactoModel, NerfactoModelConfig

from nerfstudio.cameras.camera_optimizers import CameraOptimizer, CameraOptimizerConfig


@dataclass
class NerfplayerNerfactoModelConfig(NerfactoModelConfig):
    """Nerfplayer Model Config with Nerfacto backbone"""

    _target: Type = field(default_factory=lambda: NerfplayerNerfactoModel)
    near_plane: float = 0.05
    """How far along the ray to start sampling."""
    far_plane: float = 1000.0
    """How far along the ray to stop sampling."""
    background_color: Literal["random", "last_sample", "black", "white"] = "random"
    """Whether to randomize the background color. (Random is reported to be better on DyCheck.)"""
    num_levels: int = 16
    """Hashing grid parameter."""
    features_per_level: int = 2
    """Hashing grid parameter."""
    log2_hashmap_size: int = 18
    """Hashing grid parameter."""
    temporal_dim: int = 32
    """Hashing grid parameter. A higher temporal dim means a higher temporal frequency."""
    proposal_net_args_list: List[Dict] = field(
        default_factory=lambda: [
            {"hidden_dim": 16, "temporal_dim": 32, "log2_hashmap_size": 17, "num_levels": 5, "max_res": 64},
            {"hidden_dim": 16, "temporal_dim": 32, "log2_hashmap_size": 17, "num_levels": 5, "max_res": 256},
        ]
    )
    """Arguments for the proposal density fields."""
    distortion_loss_mult: float = 1e-2
    """Distortion loss multiplier."""
    temporal_tv_weight: float = 1
    """Temporal TV balancing weight for feature channels."""
    depth_weight: float = 1e-1
    """depth loss balancing weight for feature channels."""

    camera_optimizer: CameraOptimizerConfig = field(default_factory=lambda: CameraOptimizerConfig(mode="SO3xR3"))
    """Config of the camera optimizer to use"""


class NerfplayerNerfactoModel(NerfactoModel):
    """Nerfplayer model with Nerfacto backbone.

    Args:
        config: Nerfplayer configuration to instantiate model
    """

    config: NerfplayerNerfactoModelConfig

    def populate_modules(self):
        """Set the fields and modules."""
        # Importing NerfplayerNerfactoField and TemporalHashMLPDensityField requires tcnn and CUDA
        from nerfplayer.nerfplayer_nerfacto_field import NerfplayerNerfactoField, TemporalHashMLPDensityField

        Model.populate_modules(self)

        scene_contraction = SceneContraction(order=float("inf"))

        # Fields
        self.field = NerfplayerNerfactoField(
            self.scene_box.aabb,
            temporal_dim=self.config.temporal_dim,
            num_levels=self.config.num_levels,
            features_per_level=self.config.features_per_level,
            log2_hashmap_size=self.config.log2_hashmap_size,
            spatial_distortion=scene_contraction,
            num_images=self.num_train_data,
            use_pred_normals=self.config.predict_normals,
            use_average_appearance_embedding=self.config.use_average_appearance_embedding,
        )

        self.density_fns = []
        num_prop_nets = self.config.num_proposal_iterations

        self.camera_optimizer: CameraOptimizer = self.config.camera_optimizer.setup(
            num_cameras=self.num_train_data, device="cpu"
        )

        # Build the proposal network(s)
        proposal_networks: List[TemporalHashMLPDensityField] = []
        if self.config.use_same_proposal_network:
            assert len(self.config.proposal_net_args_list) == 1, "Only one proposal network is allowed."
            prop_net_args = self.config.proposal_net_args_list[0]
            network = TemporalHashMLPDensityField(
                self.scene_box.aabb, spatial_distortion=scene_contraction, **prop_net_args
            )
            proposal_networks.append(network)
            self.density_fns.extend([network.density_fn for _ in range(num_prop_nets)])
        else:
            for i in range(num_prop_nets):
                prop_net_args = self.config.proposal_net_args_list[min(i, len(self.config.proposal_net_args_list) - 1)]
                network = TemporalHashMLPDensityField(
                    self.scene_box.aabb,
                    spatial_distortion=scene_contraction,
                    **prop_net_args,
                )
                proposal_networks.append(network)
            self.density_fns.extend([network.density_fn for network in proposal_networks])
        self.proposal_networks = cast(Sequence[TemporalHashMLPDensityField], torch.nn.ModuleList(proposal_networks))

        # Samplers
        def update_schedule(step):
            return np.clip(
                np.interp(step, [0, self.config.proposal_warmup], [0, self.config.proposal_update_every]),
                1,
                self.config.proposal_update_every,
            )

        self.proposal_sampler = ProposalNetworkSampler(
            num_nerf_samples_per_ray=self.config.num_nerf_samples_per_ray,
            num_proposal_samples_per_ray=self.config.num_proposal_samples_per_ray,
            num_proposal_network_iterations=self.config.num_proposal_iterations,
            single_jitter=self.config.use_single_jitter,
            update_sched=update_schedule,
        )

        # Collider
        self.collider = NearFarCollider(near_plane=self.config.near_plane, far_plane=self.config.far_plane)

        # renderers
        self.renderer_rgb = RGBRenderer(background_color=self.config.background_color)
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer(method="expected")  # for depth loss
        self.renderer_normals = NormalsRenderer()

        # shaders
        self.normals_shader = NormalsShader()

        # losses
        self.rgb_loss = MSELoss()

        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)
        self.temporal_distortion = True  # for viewer

    def get_outputs(self, ray_bundle: RayBundle):
        assert ray_bundle.times is not None, "Time not provided."
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler(
            ray_bundle, density_fns=[functools.partial(f, times=ray_bundle.times) for f in self.density_fns]
        )
        field_outputs = self.field(ray_samples, compute_normals=self.config.predict_normals)
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

        if self.config.predict_normals:
            outputs["normals"] = self.normals_shader(
                self.renderer_normals(normals=field_outputs[FieldHeadNames.NORMALS], weights=weights)
            )
            outputs["pred_normals"] = self.normals_shader(
                self.renderer_normals(field_outputs[FieldHeadNames.PRED_NORMALS], weights=weights)
            )

        # These use a lot of GPU memory, so we avoid storing them for eval.
        if self.training:
            outputs["weights_list"] = weights_list
            outputs["ray_samples_list"] = ray_samples_list

        if self.training and self.config.predict_normals:
            outputs["rendered_orientation_loss"] = orientation_loss(
                weights.detach(), field_outputs[FieldHeadNames.NORMALS], ray_bundle.directions
            )

            outputs["rendered_pred_normal_loss"] = pred_normal_loss(
                weights.detach(),
                field_outputs[FieldHeadNames.NORMALS].detach(),
                field_outputs[FieldHeadNames.PRED_NORMALS],
            )

        for i in range(self.config.num_proposal_iterations):
            outputs[f"prop_depth_{i}"] = self.renderer_depth(weights=weights_list[i], ray_samples=ray_samples_list[i])

        return outputs

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        loss_dict = {}
        image = batch["image"].to(self.device)
        loss_dict["rgb_loss"] = self.rgb_loss(image, outputs["rgb"])
        if self.training:
            loss_dict["interlevel_loss"] = self.config.interlevel_loss_mult * interlevel_loss(
                outputs["weights_list"], outputs["ray_samples_list"]
            )
            assert metrics_dict is not None and "distortion" in metrics_dict
            loss_dict["distortion_loss"] = self.config.distortion_loss_mult * metrics_dict["distortion"]
            if self.config.predict_normals:
                # orientation loss for computed normals
                loss_dict["orientation_loss"] = self.config.orientation_loss_mult * torch.mean(
                    outputs["rendered_orientation_loss"]
                )

                # ground truth supervision for normals
                loss_dict["pred_normal_loss"] = self.config.pred_normal_loss_mult * torch.mean(
                    outputs["rendered_pred_normal_loss"]
                )
            if "depth_image" in batch.keys() and self.config.depth_weight > 0:
                depth_image = batch["depth_image"].to(self.device)
                mask = (depth_image != 0).view([-1])
                loss_dict["depth_loss"] = 0

                def compute_depth_loss(x):
                    return self.config.depth_weight * (x - depth_image[mask]).pow(2).mean()

                loss_dict["depth_loss"] = compute_depth_loss(outputs["depth"][mask])
                for i in range(self.config.num_proposal_iterations):
                    loss_dict["depth_loss"] += compute_depth_loss(outputs[f"prop_depth_{i}"][mask])
            if self.config.temporal_tv_weight > 0:
                loss_dict["temporal_tv_loss"] = self.field.mlp_base.get_temporal_tv_loss()
                for net in self.proposal_networks:
                    loss_dict["temporal_tv_loss"] += net.encoding.get_temporal_tv_loss()
                loss_dict["temporal_tv_loss"] *= self.config.temporal_tv_weight
        return loss_dict
    

"""
nerfplayer_nerfacto_field.py
"""


from typing import Dict, Optional, Tuple

import numpy as np
import torch
from jaxtyping import Float
from torch import Tensor
from torch.nn.parameter import Parameter
import tinycudann as tcnn

from nerfstudio.cameras.rays import Frustums, RaySamples
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.field_components.activations import trunc_exp
from nerfstudio.field_components.embedding import Embedding
from nerfstudio.field_components.field_heads import (
    FieldHeadNames,
    PredNormalsFieldHead,
    SemanticFieldHead,
    TransientDensityFieldHead,
    TransientRGBFieldHead,
    UncertaintyFieldHead,
)
from nerfstudio.field_components.spatial_distortions import SpatialDistortion
from nerfstudio.fields.base_field import Field, get_normalized_directions

from nerfplayer.temporal_grid import TemporalGridEncoder


class TemporalHashMLPDensityField(Field):
    """A lightweight temporal density field module.

    Args:
        aabb: Parameters of scene aabb bounds
        temporal_dim: Hashing grid parameter. A higher temporal dim means a higher temporal frequency.
        num_layers: Number of hidden layers
        hidden_dim: Dimension of hidden layers
        spatial_distortion: Spatial distortion module
        num_levels: Hashing grid parameter. Used for initialize TemporalGridEncoder class.
        max_res: Hashing grid parameter. Used for initialize TemporalGridEncoder class.
        base_res: Hashing grid parameter. Used for initialize TemporalGridEncoder class.
        log2_hashmap_size: Hashing grid parameter. Used for initialize TemporalGridEncoder class.
        features_per_level: Hashing grid parameter. Used for initialize TemporalGridEncoder class.
    """

    def __init__(
        self,
        aabb: Tensor,
        temporal_dim: int = 64,
        num_layers: int = 2,
        hidden_dim: int = 64,
        spatial_distortion: Optional[SpatialDistortion] = None,
        num_levels: int = 8,
        max_res: int = 1024,
        base_res: int = 16,
        log2_hashmap_size: int = 18,
        features_per_level: int = 2,
    ) -> None:
        super().__init__()
        # from .temporal_grid import test; test() # DEBUG
        self.aabb = Parameter(aabb, requires_grad=False)
        self.spatial_distortion = spatial_distortion
        growth_factor = np.exp((np.log(max_res) - np.log(base_res)) / (num_levels - 1))

        self.encoding = TemporalGridEncoder(
            input_dim=3,
            temporal_dim=temporal_dim,
            num_levels=num_levels,
            level_dim=features_per_level,
            per_level_scale=growth_factor,
            base_resolution=base_res,
            log2_hashmap_size=log2_hashmap_size,
        )
        self.linear = tcnn.Network(
            n_input_dims=num_levels * features_per_level,
            n_output_dims=1,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": hidden_dim,
                "n_hidden_layers": num_layers - 1,
            },
        )

    def density_fn(
        self, positions: Float[Tensor, "*bs 3"], times: Optional[Float[Tensor, "bs 1"]] = None
    ) -> Float[Tensor, "*bs 1"]:
        """Returns only the density. Used primarily with the density grid.

        Args:
            positions: the origin of the samples/frustums
            times: the time of rays
        """
        assert times is not None, "TemporalHashMLPDensityField requires times to be specified"
        if len(positions.shape) == 3 and len(times.shape) == 2:
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

    def get_density(self, ray_samples: RaySamples) -> Tuple[Tensor, None]:
        if self.spatial_distortion is not None:
            positions = self.spatial_distortion(ray_samples.frustums.get_positions())
            positions = (positions + 2.0) / 4.0
        else:
            positions = SceneBox.get_normalized_positions(ray_samples.frustums.get_positions(), self.aabb)
        positions_flat = positions.view(-1, 3)
        assert ray_samples.times is not None
        time_flat = ray_samples.times.reshape(-1, 1)
        x = self.encoding(positions_flat, time_flat).to(positions)
        density_before_activation = self.linear(x).view(*ray_samples.frustums.shape, -1)

        # Rectifying the density with an exponential is much more stable than a ReLU or
        # softplus, because it enables high post-activation (float32) density outputs
        # from smaller internal (float16) parameters.
        density = trunc_exp(density_before_activation)
        return density, None

    def get_outputs(self, ray_samples: RaySamples, density_embedding: Optional[Tensor] = None) -> dict:
        return {}


class NerfplayerNerfactoField(Field):
    """NeRFPlayer (https://arxiv.org/abs/2210.15947) field with nerfacto backbone.

    Args:
        aabb: parameters of scene aabb bounds
        num_images: number of images in the dataset
        num_layers: number of hidden layers
        hidden_dim: dimension of hidden layers
        geo_feat_dim: output geo feat dimensions
        num_levels: number of levels of the hashmap for the base mlp
        max_res: maximum resolution of the hashmap for the base mlp
        log2_hashmap_size: size of the hashmap for the base mlp
        num_layers_color: number of hidden layers for color network
        num_layers_transient: number of hidden layers for transient network
        hidden_dim_color: dimension of hidden layers for color network
        hidden_dim_transient: dimension of hidden layers for transient network
        appearance_embedding_dim: dimension of appearance embedding
        transient_embedding_dim: dimension of transient embedding
        use_transient_embedding: whether to use transient embedding
        use_semantics: whether to use semantic segmentation
        num_semantic_classes: number of semantic classes
        use_pred_normals: whether to use predicted normals
        use_average_appearance_embedding: whether to use average appearance embedding or zeros for inference
        spatial_distortion: spatial distortion to apply to the scene
    """

    def __init__(
        self,
        aabb: Tensor,
        num_images: int,
        num_layers: int = 2,
        hidden_dim: int = 64,
        geo_feat_dim: int = 15,
        temporal_dim: int = 64,
        num_levels: int = 16,
        features_per_level: int = 2,
        log2_hashmap_size: int = 19,
        num_layers_color: int = 3,
        num_layers_transient: int = 2,
        hidden_dim_color: int = 64,
        hidden_dim_transient: int = 64,
        appearance_embedding_dim: int = 32,
        transient_embedding_dim: int = 16,
        use_transient_embedding: bool = False,
        use_semantics: bool = False,
        num_semantic_classes: int = 100,
        use_pred_normals: bool = False,
        use_average_appearance_embedding: bool = False,
        spatial_distortion: Optional[SpatialDistortion] = None,
    ) -> None:
        super().__init__()

        self.register_buffer("aabb", aabb)
        self.geo_feat_dim = geo_feat_dim

        self.spatial_distortion = spatial_distortion
        self.num_images = num_images
        self.appearance_embedding_dim = appearance_embedding_dim
        self.embedding_appearance = Embedding(self.num_images, self.appearance_embedding_dim)
        self.use_average_appearance_embedding = use_average_appearance_embedding
        self.use_transient_embedding = use_transient_embedding
        self.use_semantics = use_semantics
        self.use_pred_normals = use_pred_normals

        self.direction_encoding = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "SphericalHarmonics",
                "degree": 4,
            },
        )

        self.position_encoding = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={"otype": "Frequency", "n_frequencies": 2},
        )

        self.mlp_base = TemporalGridEncoder(
            input_dim=3,
            temporal_dim=temporal_dim,
            num_levels=num_levels,
            level_dim=features_per_level,
            log2_hashmap_size=log2_hashmap_size,
            desired_resolution=int(1024 * (self.aabb.max().item() - self.aabb.min().item())),
        )
        self.mlp_base_decode = tcnn.Network(
            n_input_dims=num_levels * features_per_level,
            n_output_dims=1 + self.geo_feat_dim,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": hidden_dim,
                "n_hidden_layers": num_layers - 1,
            },
        )

        # transients
        if self.use_transient_embedding:
            self.transient_embedding_dim = transient_embedding_dim
            self.embedding_transient = Embedding(self.num_images, self.transient_embedding_dim)
            self.mlp_transient = tcnn.Network(
                n_input_dims=self.geo_feat_dim + self.transient_embedding_dim,
                n_output_dims=hidden_dim_transient,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": hidden_dim_transient,
                    "n_hidden_layers": num_layers_transient - 1,
                },
            )
            self.field_head_transient_uncertainty = UncertaintyFieldHead(in_dim=self.mlp_transient.n_output_dims)
            self.field_head_transient_rgb = TransientRGBFieldHead(in_dim=self.mlp_transient.n_output_dims)
            self.field_head_transient_density = TransientDensityFieldHead(in_dim=self.mlp_transient.n_output_dims)

        # semantics
        if self.use_semantics:
            self.mlp_semantics = tcnn.Network(
                n_input_dims=self.geo_feat_dim,
                n_output_dims=hidden_dim_transient,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": 64,
                    "n_hidden_layers": 1,
                },
            )
            self.field_head_semantics = SemanticFieldHead(
                in_dim=self.mlp_semantics.n_output_dims, num_classes=num_semantic_classes
            )

        # predicted normals
        if self.use_pred_normals:
            self.mlp_pred_normals = tcnn.Network(
                n_input_dims=self.geo_feat_dim + self.position_encoding.n_output_dims,
                n_output_dims=hidden_dim_transient,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": 64,
                    "n_hidden_layers": 2,
                },
            )
            self.field_head_pred_normals = PredNormalsFieldHead(in_dim=self.mlp_pred_normals.n_output_dims)

        self.mlp_head = tcnn.Network(
            n_input_dims=self.direction_encoding.n_output_dims + self.geo_feat_dim + self.appearance_embedding_dim,
            n_output_dims=3,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "Sigmoid",
                "n_neurons": hidden_dim_color,
                "n_hidden_layers": num_layers_color - 1,
            },
        )

    def get_density(self, ray_samples: RaySamples) -> Tuple[Tensor, Tensor]:
        """Computes and returns the densities."""
        if self.spatial_distortion is not None:
            positions = ray_samples.frustums.get_positions()
            positions = self.spatial_distortion(positions)
            positions = (positions + 2.0) / 4.0
        else:
            positions = SceneBox.get_normalized_positions(ray_samples.frustums.get_positions(), self.aabb)
        positions_flat = positions.view(-1, 3)
        assert ray_samples.times is not None, "Time should be included in the input for NeRFPlayer"
        time_flat = ray_samples.times.reshape(-1, 1)
        h = self.mlp_base(positions_flat, time_flat)
        h = self.mlp_base_decode(h).view(*ray_samples.frustums.shape, -1)
        density_before_activation, base_mlp_out = torch.split(h, [1, self.geo_feat_dim], dim=-1)

        # Rectifying the density with an exponential is much more stable than a ReLU or
        # softplus, because it enables high post-activation (float32) density outputs
        # from smaller internal (float16) parameters.
        density = trunc_exp(density_before_activation.to(positions))
        return density, base_mlp_out

    def get_outputs(
        self, ray_samples: RaySamples, density_embedding: Optional[Tensor] = None
    ) -> Dict[FieldHeadNames, Tensor]:
        assert density_embedding is not None
        outputs = {}
        if ray_samples.camera_indices is None:
            raise AttributeError("Camera indices are not provided.")
        camera_indices = ray_samples.camera_indices.squeeze()
        directions = get_normalized_directions(ray_samples.frustums.directions)
        directions_flat = directions.view(-1, 3)
        d = self.direction_encoding(directions_flat)

        outputs_shape = ray_samples.frustums.directions.shape[:-1]

        # appearance
        if self.training:
            embedded_appearance = self.embedding_appearance(camera_indices)
        else:
            if self.use_average_appearance_embedding:
                embedded_appearance = torch.ones(
                    (*directions.shape[:-1], self.appearance_embedding_dim), device=directions.device
                ) * self.embedding_appearance.mean(dim=0)
            else:
                embedded_appearance = torch.zeros(
                    (*directions.shape[:-1], self.appearance_embedding_dim), device=directions.device
                )

        # transients
        if self.use_transient_embedding and self.training:
            embedded_transient = self.embedding_transient(camera_indices)
            transient_input = torch.cat(
                [
                    density_embedding.view(-1, self.geo_feat_dim),
                    embedded_transient.view(-1, self.transient_embedding_dim),
                ],
                dim=-1,
            )
            x = self.mlp_transient(transient_input).view(*outputs_shape, -1).to(directions)
            outputs[FieldHeadNames.UNCERTAINTY] = self.field_head_transient_uncertainty(x)
            outputs[FieldHeadNames.TRANSIENT_RGB] = self.field_head_transient_rgb(x)
            outputs[FieldHeadNames.TRANSIENT_DENSITY] = self.field_head_transient_density(x)

        # semantics
        if self.use_semantics:
            density_embedding_copy = density_embedding.clone().detach()
            semantics_input = torch.cat(
                [
                    density_embedding_copy.view(-1, self.geo_feat_dim),
                ],
                dim=-1,
            )
            x = self.mlp_semantics(semantics_input).view(*outputs_shape, -1).to(directions)
            outputs[FieldHeadNames.SEMANTICS] = self.field_head_semantics(x)

        # predicted normals
        if self.use_pred_normals:
            positions = ray_samples.frustums.get_positions()

            positions_flat = self.position_encoding(positions.view(-1, 3))
            pred_normals_inp = torch.cat([positions_flat, density_embedding.view(-1, self.geo_feat_dim)], dim=-1)

            x = self.mlp_pred_normals(pred_normals_inp).view(*outputs_shape, -1).to(directions)
            outputs[FieldHeadNames.PRED_NORMALS] = self.field_head_pred_normals(x)

        h = torch.cat(
            [
                d,
                density_embedding.view(-1, self.geo_feat_dim),
                embedded_appearance.view(-1, self.appearance_embedding_dim),
            ],
            dim=-1,
        )
        rgb = self.mlp_head(h).view(*outputs_shape, -1).to(directions)
        outputs.update({FieldHeadNames.RGB: rgb})

        return outputs
    

"""
nerfplayer_ngp.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Type, TYPE_CHECKING

import nerfacc
import torch
from torch.nn import Parameter
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.model_components.losses import MSELoss
from nerfstudio.model_components.ray_samplers import VolumetricSampler
from nerfstudio.model_components.renderers import (
    AccumulationRenderer,
    DepthRenderer,
    RGBRenderer,
)
from nerfstudio.models.base_model import Model
from nerfstudio.models.instant_ngp import InstantNGPModelConfig, NGPModel
if TYPE_CHECKING:
    # Importing NerfplayerNGPField requires tcnn and CUDA
    from nerfplayer.nerfplayer_ngp_field import NerfplayerNGPField


@dataclass
class NerfplayerNGPModelConfig(InstantNGPModelConfig):
    """NeRFPlayer Model Config with InstantNGP backbone.
    Tips for tuning the performance:
    1. If the scene is flickering, this is caused by unwanted high-freq on the temporal dimension.
        Try reducing `temporal_dim` first, but don't be too small, otherwise the dynamic object is blurred.
        Then try increasing the `temporal_tv_weight`. This is the loss for promoting smoothness among the
        temporal channels.
    2. If a faster rendering is preferred, then try reducing `log2_hashmap_size`. If more details are
        wanted, try increasing `log2_hashmap_size`.
    3. If the input cameras are of limited numbers, try reducing `num_levels`. `num_levels` is for
        multi-resolution volume sampling, and has a similar behavior to the freq in NeRF. With a small
        `num_levels`, a blurred rendering will be generated, but it is unlikely to overfit the training views.
    """

    _target: Type = field(default_factory=lambda: NerfplayerNGPModel)
    temporal_dim: int = 64
    """Hashing grid parameter. A higher temporal dim means a higher temporal frequency."""
    num_levels: int = 16
    """Hashing grid parameter."""
    features_per_level: int = 2
    """Hashing grid parameter."""
    log2_hashmap_size: int = 17
    """Hashing grid parameter."""
    base_resolution: int = 16
    """Hashing grid parameter."""
    temporal_tv_weight: float = 1
    """Temporal TV loss balancing weight for feature channels."""
    depth_weight: float = 1e-1
    """depth loss balancing weight for feature channels."""
    train_background_color: Literal["random", "black", "white"] = "random"
    """The training background color that is given to untrained areas."""
    eval_background_color: Literal["random", "black", "white"] = "white"
    """The training background color that is given to untrained areas."""
    disable_viewing_dependent: bool = True
    """Disable viewing dependent effects."""
    disable_scene_contraction: bool = False
    """Whether to disable scene contraction or not."""


class NerfplayerNGPModel(NGPModel):
    """NeRFPlayer Model with InstantNGP backbone.

    Args:
        config: NeRFPlayer NGP configuration to instantiate model
    """

    config: NerfplayerNGPModelConfig
    field: 'NerfplayerNGPField'

    def populate_modules(self):
        """Set the fields and modules."""
        # Importing NerfplayerNGPField requires tcnn and CUDA
        from nerfplayer.nerfplayer_ngp_field import NerfplayerNGPField

        Model.populate_modules(self)

        if self.config.disable_scene_contraction:
            scene_contraction = None
        else:
            scene_contraction = SceneContraction(order=float("inf"))

        self.field = NerfplayerNGPField(
            aabb=self.scene_box.aabb,
            use_appearance_embedding=self.config.use_appearance_embedding,
            num_images=self.num_train_data,
            temporal_dim=self.config.temporal_dim,
            num_levels=self.config.num_levels,
            features_per_level=self.config.features_per_level,
            log2_hashmap_size=self.config.log2_hashmap_size,
            base_resolution=self.config.base_resolution,
            disable_viewing_dependent=self.config.disable_viewing_dependent,
            spatial_distortion=scene_contraction,
        )

        self.scene_aabb = Parameter(self.scene_box.aabb.flatten(), requires_grad=False)

        # Occupancy Grid
        self.occupancy_grid = nerfacc.OccGridEstimator(
            roi_aabb=self.scene_aabb,
            resolution=self.config.grid_resolution,
            levels=self.config.grid_levels,
        )

        # Sampler
        self.sampler = VolumetricSampler(
            occupancy_grid=self.occupancy_grid,
            density_fn=self.field.density_fn,
        )  # need to update the density_fn later during forward (for input time)

        # renderers
        self.renderer_rgb = RGBRenderer()  # will update bgcolor later during forward
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer(method="expected")

        # losses
        self.rgb_loss = MSELoss()

        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)
        self.temporal_distortion = True  # for viewer

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        def update_occupancy_grid(step: int):
            self.occupancy_grid.update_every_n_steps(
                step=step,
                occ_eval_fn=lambda x: self.field.get_opacity(x, self.config.render_step_size),
            )

        return [
            TrainingCallback(
                where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                update_every_num_iters=1,
                func=update_occupancy_grid,
            ),
        ]

    def get_outputs(self, ray_bundle: RayBundle):
        num_rays = len(ray_bundle)

        with torch.no_grad():
            ray_samples, ray_indices = self.sampler(
                ray_bundle=ray_bundle,
                near_plane=self.config.near_plane,
                far_plane=self.config.far_plane,
                render_step_size=self.config.render_step_size,
                cone_angle=self.config.cone_angle,
                alpha_thre=self.config.alpha_thre,
            )

        field_outputs = self.field(ray_samples)

        # accumulation
        packed_info = nerfacc.pack_info(ray_indices, num_rays)
        weights = nerfacc.render_weight_from_density(
            packed_info=packed_info,
            sigmas=field_outputs[FieldHeadNames.DENSITY][..., 0],
            t_starts=ray_samples.frustums.starts[..., 0],
            t_ends=ray_samples.frustums.ends[..., 0],
        )[0]
        weights = weights[..., None]

        # update bgcolor in the renderer; usually random color for training and fixed color for inference
        if self.training:
            self.renderer_rgb.background_color = self.config.train_background_color
        else:
            self.renderer_rgb.background_color = self.config.eval_background_color
        rgb = self.renderer_rgb(
            rgb=field_outputs[FieldHeadNames.RGB],
            weights=weights,
            ray_indices=ray_indices,
            num_rays=num_rays,
        )
        depth = self.renderer_depth(
            weights=weights, ray_samples=ray_samples, ray_indices=ray_indices, num_rays=num_rays
        )
        accumulation = self.renderer_accumulation(weights=weights, ray_indices=ray_indices, num_rays=num_rays)

        outputs = {
            "rgb": rgb,
            "accumulation": accumulation,
            "depth": depth,
            "num_samples_per_ray": packed_info[:, 1],
        }
        # adding them to outputs for calculating losses
        if self.training and self.config.depth_weight > 0:
            outputs["ray_indices"] = ray_indices
            outputs["ray_samples"] = ray_samples
            outputs["weights"] = weights
            outputs["sigmas"] = field_outputs[FieldHeadNames.DENSITY]
        return outputs

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        image = batch["image"].to(self.device)
        rgb_loss = self.rgb_loss(image, outputs["rgb"])
        loss_dict = {"rgb_loss": rgb_loss}
        if "depth_image" in batch.keys() and self.config.depth_weight > 0 and self.training:
            depth_image = batch["depth_image"].to(self.device)
            mask = depth_image != 0
            # First we calculate the depth value, just like most of the papers.
            loss_dict["depth_loss"] = (outputs["depth"][mask] - depth_image[mask]).abs().mean()
            # But this is not enough -- it will lead to fog like reconstructions, even with depth supervision.
            # Because this loss only cares about the mean value.
            # (Feel free to try it and see the fog-like effects on DyCheck by commenting out the following loss.)
            # > The fog can be effectively penalized by multiview inputs (i.e., from another viewing point).
            # > However, it is hard to get multiple views under the setting of dynamic scenes.
            # > A new view always indicates another camera, which is expensive and brings sync problems.
            # In DyCheck (https://arxiv.org/abs/2210.13445), surface sparsity regularizer
            # (i.e., `distortion_loss` in nerfstudio) is used to make surface tight.
            # The `distortion_loss` can be used here as well, but personally find it hard to implement...
            # (Due to volume sampling, seems that cuda kernels are needed for efficiently computing the loss.)
            # (Try nerfplayer with nerfacto backbone for distortion loss.)
            # Instead, directly penalizing the empty space is found effective here. Perhaps due to a more
            # "clear" loss (as it is directly applied to the network outputs, rather than post-processed values).
            # But such a loss also has drawbacks: it tends to overfit wrong (or noise) presented in the depth map.
            # Some structures are floating in the air with this loss...
            gt_depth_packed = depth_image[outputs["ray_indices"]]
            steps = (outputs["ray_samples"].frustums.starts + outputs["ray_samples"].frustums.ends) / 2
            # empty area should not be too close to the given depth, so lets add a margin to the gt depth
            margin = (self.scene_box.aabb.max() - self.scene_box.aabb.min()) / 128
            density_min_mask = (gt_depth_packed - steps > margin) & (gt_depth_packed != 0)
            loss_dict["depth_loss"] += (outputs["sigmas"][density_min_mask[..., 0]].pow(2)).mean() * 1e-2
            loss_dict["depth_loss"] *= self.config.depth_weight
        if self.config.temporal_tv_weight > 0 and self.training:
            loss_dict["temporal_tv_loss"] = self.config.temporal_tv_weight * self.field.mlp_base.get_temporal_tv_loss()
        return loss_dict
    

"""
nerfplayer_ngp_field.py
"""


from typing import Dict, Optional, Tuple

import torch
from jaxtyping import Float
from torch import Tensor
from torch.nn.parameter import Parameter
import tinycudann as tcnn

from nerfstudio.cameras.rays import Frustums, RaySamples
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.field_components.activations import trunc_exp
from nerfstudio.field_components.embedding import Embedding
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import (
    SceneContraction,
    SpatialDistortion,
)
from nerfstudio.fields.base_field import Field, get_normalized_directions

from nerfplayer.temporal_grid import TemporalGridEncoder


class NerfplayerNGPField(Field):
    """NeRFPlayer (https://arxiv.org/abs/2210.15947) field with InstantNGP backbone.

    Args:
        aabb: parameters of scene aabb bounds
        temporal_dim: the dimension of temporal axis, a higher dimension indicates a higher temporal frequency
            please refer to the implementation of TemporalGridEncoder for more details
        num_levels: the number of multi-resolution levels; same as InstantNGP
        features_per_level: the dim of output feature vector for each level; same as InstantNGP
        log2_hashmap_size: the size of the table; same as InstantNGP
        base_resolution: base resolution for the table; same as InstantNGP
        num_layers: number of hidden layers (occupancy decoder network after sampling)
        hidden_dim: dimension of hidden layers (occupancy decoder network after sampling)
        geo_feat_dim: output geo feat dimensions
        num_layers_color: number of hidden layers for color network
        hidden_dim_color: dimension of hidden layers for color network
        use_appearance_embedding: whether to use appearance embedding
        disable_viewing_dependent: if true, disable the viewing dependent effect (no viewing direction as inputs)
            Sometimes we need to disable viewing dependent effects in a dynamic scene, because there is
            ambiguity between being dynamic and viewing dependent effects. For example, the shadow of the camera
            should be a dynamic effect, but may be reconstructed as viewing dependent effects.
        num_images: number of images, required if use_appearance_embedding is True
        appearance_embedding_dim: dimension of appearance embedding
        contraction_type: type of contraction
    """

    def __init__(
        self,
        aabb: Tensor,
        temporal_dim: int = 16,
        num_levels: int = 16,
        features_per_level: int = 2,
        log2_hashmap_size: int = 19,
        base_resolution: int = 16,
        num_layers: int = 2,
        hidden_dim: int = 64,
        geo_feat_dim: int = 15,
        num_layers_color: int = 3,
        hidden_dim_color: int = 64,
        use_appearance_embedding: bool = False,
        disable_viewing_dependent: bool = False,
        num_images: Optional[int] = None,
        appearance_embedding_dim: int = 32,
        spatial_distortion: Optional[SpatialDistortion] = SceneContraction(),
    ) -> None:
        super().__init__()

        self.aabb = Parameter(aabb, requires_grad=False)
        self.geo_feat_dim = geo_feat_dim
        self.spatial_distortion = spatial_distortion

        self.use_appearance_embedding = use_appearance_embedding
        if use_appearance_embedding:
            assert num_images is not None
            self.appearance_embedding_dim = appearance_embedding_dim
            self.appearance_embedding = Embedding(num_images, appearance_embedding_dim)

        self.direction_encoding = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "SphericalHarmonics",
                "degree": 4,
            },
        )

        self.mlp_base = TemporalGridEncoder(
            input_dim=3,
            temporal_dim=temporal_dim,
            num_levels=num_levels,
            level_dim=features_per_level,
            base_resolution=base_resolution,
            log2_hashmap_size=log2_hashmap_size,
            desired_resolution=int(1024 * (self.aabb.max() - self.aabb.min())),
        )
        self.mlp_base_decode = tcnn.Network(
            n_input_dims=num_levels * features_per_level,
            n_output_dims=1 + self.geo_feat_dim,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": hidden_dim,
                "n_hidden_layers": num_layers - 1,
            },
        )

        in_dim = self.direction_encoding.n_output_dims + self.geo_feat_dim
        if disable_viewing_dependent:
            in_dim = self.geo_feat_dim
            self.direction_encoding = None
        if self.use_appearance_embedding:
            in_dim += self.appearance_embedding_dim
        self.mlp_head = tcnn.Network(
            n_input_dims=in_dim,
            n_output_dims=3,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "Sigmoid",
                "n_neurons": hidden_dim_color,
                "n_hidden_layers": num_layers_color - 1,
            },
        )

    def get_density(self, ray_samples: RaySamples) -> Tuple[Tensor, Tensor]:
        if self.spatial_distortion is not None:
            positions = ray_samples.frustums.get_positions()
            positions = self.spatial_distortion(positions)
            positions = (positions + 2.0) / 4.0
        else:
            positions = SceneBox.get_normalized_positions(ray_samples.frustums.get_positions(), self.aabb)
        # Make sure the tcnn gets inputs between 0 and 1.
        selector = ((positions > 0.0) & (positions < 1.0)).all(dim=-1)
        positions = positions * selector[..., None]
        positions_flat = positions.view(-1, 3)
        assert ray_samples.times is not None, "Time should be included in the input for NeRFPlayer"
        times_flat = ray_samples.times.view(-1, 1)

        h = self.mlp_base(positions_flat, times_flat)
        h = self.mlp_base_decode(h).view(*ray_samples.frustums.shape, -1)
        density_before_activation, base_mlp_out = torch.split(h, [1, self.geo_feat_dim], dim=-1)

        # Rectifying the density with an exponential is much more stable than a ReLU or
        # softplus, because it enables high post-activation (float32) density outputs
        # from smaller internal (float16) parameters.
        density = trunc_exp(density_before_activation.to(positions))
        density = density * selector[..., None]
        return density, base_mlp_out

    def get_outputs(
        self, ray_samples: RaySamples, density_embedding: Optional[Tensor] = None
    ) -> Dict[FieldHeadNames, Tensor]:
        assert density_embedding is not None
        directions = get_normalized_directions(ray_samples.frustums.directions)
        directions_flat = directions.view(-1, 3)

        if self.direction_encoding is not None:
            d = self.direction_encoding(directions_flat)
            h = torch.cat([d, density_embedding.view(-1, self.geo_feat_dim)], dim=-1)
        else:
            # viewing direction is disabled
            h = density_embedding.view(-1, self.geo_feat_dim)

        if self.use_appearance_embedding:
            if ray_samples.camera_indices is None:
                raise AttributeError("Camera indices are not provided.")
            camera_indices = ray_samples.camera_indices.squeeze()
            if self.training:
                embedded_appearance = self.appearance_embedding(camera_indices)
            else:
                embedded_appearance = torch.zeros(
                    (*directions.shape[:-1], self.appearance_embedding_dim), device=directions.device
                )
            h = torch.cat([h, embedded_appearance.view(-1, self.appearance_embedding_dim)], dim=-1)

        rgb = self.mlp_head(h).view(*ray_samples.frustums.directions.shape[:-1], -1).to(directions)
        return {FieldHeadNames.RGB: rgb}

    def density_fn(
        self, positions: Float[Tensor, "*bs 3"], times: Optional[Float[Tensor, "*bs 1"]] = None
    ) -> Float[Tensor, "*bs 1"]:
        """Returns only the density. Used primarily with the density grid.
        Overwrite this function since density is time dependent now.

        Args:
            positions: the origin of the samples/frustums
            times: the time of each position
        """
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

    def get_opacity(self, positions: Float[Tensor, "*bs 3"], step_size, time_intervals=10) -> Float[Tensor, "*bs 1"]:
        """Returns the opacity for a position and time. Used primarily by the occupancy grid.
        This will return the maximum opacity of the points in the space in a dynamic sequence.

        Args:
            positions: the positions to evaluate the opacity at.
            step_size: the step size to use for the opacity evaluation.
            time_intervals: sample density on N time stamps
        """
        # TODO: Converting opacity by time intervals is slow, and may lead to temporal artifacts.
        #       (Maybe random sample time and EMA?)
        opacity = []
        for t in range(0, time_intervals):
            density = self.density_fn(positions, t / (time_intervals - 1) * torch.ones_like(positions)[..., [0]])
            opacity.append(density * step_size)
        opacity = torch.stack(opacity, dim=0).max(dim=0).values
        return opacity
    



"""Implements the temporal grid used by NeRFPlayer (https://arxiv.org/abs/2210.15947).
A time conditioned sliding window is applied on the feature channels, so
that the feature vectors become time-aware.
temporal_grid.py
"""
from typing import Optional

import numpy as np
import torch
from jaxtyping import Float, Int
from torch import Tensor, nn
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd

import nerfplayer.cuda as _C


class TemporalGridEncodeFunc(Function):
    """Class for autograd in pytorch."""

    @staticmethod
    @custom_fwd
    def forward(
        ctx,
        inputs: Float[Tensor, "bs input_dim"],
        temporal_row_index: Float[Tensor, "bs temporal_index_dim"],
        embeddings: Float[Tensor, "table_size embed_dim"],
        offsets: Int[Tensor, "num_levels_plus_1"],
        per_level_scale: float,
        base_resolution: int,
        calc_grad_inputs: bool = False,
        gridtype: int = 0,
        align_corners: bool = False,
    ) -> Float[Tensor, "bs output_dim"]:
        """Call forward and interpolate the feature from embeddings

        Args:
            inputs: the input coords
            temporal_row_index: the input index of channels for doing the interpolation
            embeddings: the saved (hashing) table for the feature grid (of the full sequence)
            offsets: offsets for each level in the multilevel table, used for locating in cuda kernels
            per_level_scale: scale parameter for the table; same as InstantNGP
            base_resolution: base resolution for the table; same as InstantNGP
            calc_grad_inputs: bool indicator for calculating gradients on the inputs
            gridtype: 0 == hash, 1 == tiled; tiled is a baseline in InstantNGP (not random collision)
            align_corners: same as other interpolation operators
        """

        inputs = inputs.contiguous()
        temporal_row_index = temporal_row_index.contiguous()

        B, D = inputs.shape  # batch size, coord dim
        L = offsets.shape[0] - 1  # level
        grid_channel = embeddings.shape[1]  # embedding dim for each level
        C = temporal_row_index.shape[1] // 4  # output embedding dim for each level
        S = np.log2(per_level_scale)  # resolution multiplier at each level, apply log2 for later CUDA exp2f
        H = base_resolution  # base resolution

        # torch.half used by torch-ngp, but we disable it
        # (could be of negative impact on the performance? not sure, but feel free to inform me and help improve it!)
        # # manually handle autocast (only use half precision embeddings, inputs must be float for enough precision)
        # # if C % 2 != 0, force float, since half for atomicAdd is very slow.
        # if torch.is_autocast_enabled() and C % 2 == 0:
        #     embeddings = embeddings.to(torch.half)

        # L first, optimize cache for cuda kernel, but needs an extra permute later
        outputs = torch.empty(L, B, C, device=inputs.device, dtype=embeddings.dtype)

        if calc_grad_inputs:
            dy_dx = torch.empty(B, L * D * C, device=inputs.device, dtype=embeddings.dtype)
        else:
            dy_dx = None

        _C.temporal_grid_encode_forward(
            inputs,
            temporal_row_index,
            embeddings,
            offsets,
            outputs,
            B,
            D,
            grid_channel,
            C,
            L,
            S,
            H,
            dy_dx,
            gridtype,
            align_corners,
        )

        # permute back to [B, L * C]
        outputs = outputs.permute(1, 0, 2).reshape(B, L * C)

        ctx.save_for_backward(inputs, temporal_row_index, embeddings, offsets, dy_dx)
        ctx.dims = [B, D, grid_channel, C, L, S, H, gridtype]
        ctx.align_corners = align_corners

        return outputs

    @staticmethod
    @custom_bwd
    def backward(ctx, grad):
        inputs, temporal_row_index, embeddings, offsets, dy_dx = ctx.saved_tensors
        B, D, grid_channel, C, L, S, H, gridtype = ctx.dims
        align_corners = ctx.align_corners

        # grad: [B, L * C] --> [L, B, C]
        grad = grad.view(B, L, C).permute(1, 0, 2).contiguous()

        grad_embeddings = torch.zeros_like(embeddings).contiguous()

        if dy_dx is not None:
            grad_inputs = torch.zeros_like(inputs, dtype=embeddings.dtype)
        else:
            grad_inputs = None

        _C.temporal_grid_encode_backward(
            grad,
            inputs,
            temporal_row_index,
            embeddings,
            offsets,
            grad_embeddings,
            B,
            D,
            grid_channel,
            C,
            L,
            S,
            H,
            dy_dx,
            grad_inputs,
            gridtype,
            align_corners,
        )

        if grad_inputs is not None:
            grad_inputs = grad_inputs.to(inputs.dtype)

        return grad_inputs, None, grad_embeddings, None, None, None, None, None, None


class TemporalGridEncoder(nn.Module):
    """Class for temporal grid encoding.
    This class extends the grid encoding (from InstantNGP) by allowing the output time-dependent feature channels.
    For example, for time 0 the interpolation uses channels [0,1], then for time 1 channels [2,1] are used.
    This operation can be viewed as applying a time-dependent sliding window on the feature channels.

    Args:
        temporal_dim: the dimension of temporal modeling; a higher dim indicates a higher freq on the time axis
        input_dim: the dimension of input coords
        num_levels: number of levels for multi-scale hashing; same as InstantNGP
        level_dim: the dim of output feature vector for each level; same as InstantNGP
        per_level_scale: scale factor; same as InstantNGP
        base_resolution: base resolution for the table; same as InstantNGP
        log2_hashmap_size: the size of the table; same as InstantNGP
        desired_resolution: desired resolution at the last level; same as InstantNGP
        gridtype: "tiled" or "hash"
        align_corners: same as other interpolation operators
    """

    sampling_index: Tensor
    index_a_mask: Tensor
    index_b_mask: Tensor
    index_list: Tensor

    def __init__(
        self,
        temporal_dim: int = 64,
        input_dim: int = 3,
        num_levels: int = 16,
        level_dim: int = 2,
        per_level_scale: float = 2.0,
        base_resolution: int = 16,
        log2_hashmap_size: int = 19,
        desired_resolution: Optional[int] = None,
        gridtype: str = "hash",
        align_corners: bool = False,
    ) -> None:
        super().__init__()

        # the finest resolution desired at the last level, if provided, overridee per_level_scale
        if desired_resolution is not None:
            per_level_scale = np.exp2(np.log2(desired_resolution / base_resolution) / (num_levels - 1))

        self.temporal_dim = temporal_dim
        self.input_dim = input_dim  # coord dims, 2 or 3
        self.num_levels = num_levels  # num levels, each level multiply resolution by 2
        self.level_dim = level_dim  # encode channels per level
        self.per_level_scale = per_level_scale  # multiply resolution by this scale at each level.
        self.log2_hashmap_size = log2_hashmap_size
        self.base_resolution = base_resolution
        self.output_dim = num_levels * level_dim
        self.gridtype = gridtype
        _gridtype_to_id = {"hash": 0, "tiled": 1}
        self.gridtype_id = _gridtype_to_id[gridtype]  # "tiled" or "hash"
        self.align_corners = align_corners

        # allocate parameters
        offsets = []
        offset = 0
        self.max_params = 2**log2_hashmap_size
        for i in range(num_levels):
            resolution = int(np.ceil(base_resolution * per_level_scale**i))
            params_in_level = min(
                self.max_params, (resolution if align_corners else resolution + 1) ** input_dim
            )  # limit max number
            params_in_level = int(np.ceil(params_in_level / 8) * 8)  # make divisible
            offsets.append(offset)
            offset += params_in_level
        offsets.append(offset)
        offsets = torch.from_numpy(np.array(offsets, dtype=np.int32))
        self.register_buffer("offsets", offsets)
        self.n_params = offsets[-1] * level_dim
        self.embeddings = nn.Parameter(torch.empty(offset, level_dim + temporal_dim))
        self.init_parameters()

    def init_parameters(self) -> None:
        """Initialize the parameters:
        1. Uniform initialization of the embeddings
        2. Temporal interpolation index initialization:
            For each temporal dim, we initialize a interpolation candidate.
            For example, if temporal dim 0, we use channels [0,1,2,3], then for temporal dim 1,
            we use channels [4,1,2,3]. After that, temporal dim 2, we use channels [4,5,2,3].
            This is for the alignment of the channels. I.e., each temporal dim should differ
            on only one channel, otherwise moving from one temporal dim to the next one is not
            that consistent.
            To associate time w.r.t. temporal dim, we evenly distribute time into the temporal dims.
            That is, if we have 16 temporal dims, then the 16th channel combinations is the time 1.
            (Time should be within 0 and 1.) Given a time, we first look up which temporal dim should
            be used. And then compute the linear combination weights.
            For implementing it, a table for all possible channel combination are used. Each row in
            the table is the candidate feature channels, and means we move from one temporal dim to
            the next one. For example, the first row will use feature channels [0,1,2,3,4]. Each row
            is of length `num_of_output_channel*4`. The expanding param 4 is for saving the combination
            weights and channels. The first row will be [?,0,?,1, 1,2,0,0, 1,3,0,0, 1,4,0,0]. Each
            4 tuple means
                `[weight_for_channel_A, index_for_channel_A, weight_for_channel_B, index_for_channel_B]`
            If `weight_for_channel_A` is 1, then there is no interpolation on this channel.
        """
        std = 1e-4
        self.embeddings.data.uniform_(-std, std)
        # generate sampling index
        temporal_grid_rows = self.temporal_dim
        index_init = [0, self.level_dim] + list(range(1, self.level_dim))
        permute_base = list(range(2, self.level_dim + 1))
        last_entry = 0  # insert into ith place
        permute_init = permute_base[:last_entry] + [0] + permute_base[last_entry:]
        index_list = [torch.as_tensor(index_init, dtype=torch.long)]
        permute_list = [torch.as_tensor(permute_init, dtype=torch.long)]

        # converts a list of channel candidates into sampling row
        def to_sampling_index(index, permute, last_entry):
            row = index[permute]
            row = torch.stack([torch.ones_like(row), row, torch.zeros_like(row), torch.zeros_like(row)], 1)
            row = row.reshape([-1])
            mask_a = torch.zeros_like(row).bool()
            mask_b = torch.zeros_like(row).bool()
            row[last_entry * 4 + 3] = index[1]
            mask_a[last_entry * 4] = 1
            mask_b[last_entry * 4 + 2] = 1
            return row, mask_a, mask_b

        row, mask_a, mask_b = to_sampling_index(index_list[0], permute_list[0], last_entry)
        sampling_index = [row]
        index_a_mask, index_b_mask = [mask_a], [mask_b]
        # iterate on all temporal grid to get all rows
        for _ in range(1, temporal_grid_rows - 1):
            # the following lines are a little confusing...
            # the basic idea is to keep a buffer and then move to the next channel
            last_entry += 1
            if last_entry >= self.level_dim:
                last_entry = 0
            last_index_max = index_list[-1].max().item()
            last_index_min = index_list[-1].min().item()
            tem_permute_list = permute_list[-1].clone()  # for rearrange
            tem_permute_list[tem_permute_list == 0] += 1
            prev = index_list[-1][1:][tem_permute_list - 1].tolist()
            prev.pop(last_entry)
            new_index = [last_index_min + 1, last_index_max + 1] + prev
            new_index = torch.as_tensor(new_index, dtype=torch.long)
            new_permute = permute_base[:last_entry] + [0] + permute_base[last_entry:]
            new_permute = torch.as_tensor(new_permute, dtype=torch.long)
            index_list.append(torch.as_tensor(new_index, dtype=torch.long))
            permute_list.append(torch.as_tensor(new_permute, dtype=torch.long))
            row, mask_a, mask_b = to_sampling_index(index_list[-1], permute_list[-1], last_entry)
            sampling_index.append(row)
            index_a_mask.append(mask_a)
            index_b_mask.append(mask_b)
        self.register_buffer("index_list", torch.stack(index_list))
        self.register_buffer("sampling_index", torch.stack(sampling_index))
        # index_a_mask and index_b_mask are for inserting the combination weights
        self.register_buffer("index_a_mask", torch.stack(index_a_mask))
        self.register_buffer("index_b_mask", torch.stack(index_b_mask))

    def __repr__(self) -> str:
        """For debug and logging purpose."""
        return (
            f"GridEncoder: input_dim={self.input_dim} num_levels={self.num_levels} "
            f"level_dim={self.level_dim} resolution={self.base_resolution} -> "
            f"{int(round(self.base_resolution * self.per_level_scale ** (self.num_levels - 1)))} "
            f"per_level_scale={self.per_level_scale:.4f} params={tuple(self.embeddings.shape)} "
            f"gridtype={self.gridtype} align_corners={self.align_corners}"
        )

    def get_temporal_index(self, time: Float[Tensor, "bs"]) -> Float[Tensor, "bs temporal_index_dim"]:
        """Convert the time into sampling index lists."""
        row_idx_value = time * (len(self.sampling_index) - 1)
        row_idx = row_idx_value.long()
        row_idx[time == 1] = len(self.sampling_index) - 1
        temporal_row_index = self.sampling_index[row_idx].float()
        mask_a = self.index_a_mask[row_idx]
        mask_b = self.index_b_mask[row_idx]
        temporal_row_index[mask_a] = row_idx + 1 - row_idx_value
        temporal_row_index[mask_b] = row_idx_value - row_idx
        return temporal_row_index

    def forward(
        self, xyz: Float[Tensor, "bs input_dim"], time: Float[Tensor, "bs 1"]
    ) -> Float[Tensor, "bs output_dim"]:
        """Forward and sampling feature vectors from the embedding.

        Args:
            xyz: input coords, should be in [0,1]
            time: input time, should be in [0,1] with shape [bs, 1]
        """
        outputs = TemporalGridEncodeFunc.apply(
            xyz,
            self.get_temporal_index(time[:, 0].float()),
            self.embeddings,
            self.offsets,
            self.per_level_scale,
            self.base_resolution,
            xyz.requires_grad,
            self.gridtype_id,
            self.align_corners,
        )
        assert isinstance(outputs, Tensor)
        return outputs

    def get_temporal_tv_loss(self) -> Float[Tensor, ""]:
        """Apply TV loss on the temporal channels.
        Sample a random channel combination (i.e., row for the combination table),
        and then compute loss on it.
        """
        row_idx = torch.randint(0, len(self.index_list), [1]).item()
        feat_idx = self.index_list[row_idx]
        return (self.embeddings[:, feat_idx[0]] - self.embeddings[:, feat_idx[1]]).abs().mean()