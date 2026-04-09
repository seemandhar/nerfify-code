"seathru_config.py"

from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.plugins.types import MethodSpecification
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManagerConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.configs.base_config import ViewerConfig

from seathru.seathru_model import SeathruModelConfig


# Base method configuration
seathru_method = MethodSpecification(
    config=TrainerConfig(
        method_name="seathru-nerf",
        steps_per_eval_batch=500,
        steps_per_save=2000,
        max_num_iterations=100000,
        mixed_precision=True,
        pipeline=VanillaPipelineConfig(
            datamanager=VanillaDataManagerConfig(
                dataparser=NerfstudioDataParserConfig(),
                train_num_rays_per_batch=16384,
                eval_num_rays_per_batch=4096,
            ),
            model=SeathruModelConfig(eval_num_rays_per_chunk=1 << 15),
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=2e-3, eps=1e-8),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1e-5, max_steps=500000, warmup_steps=1024
                ),
            },
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=2e-3, eps=1e-8, max_norm=0.001),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1e-5, max_steps=500000, warmup_steps=1024
                ),
            },
            "camera_opt": {
                "mode": "off",
                "optimizer": AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=6e-6, max_steps=500000
                ),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="SeaThru-NeRF for underwater scenes.",
)

# Lite method configuration
seathru_method_lite = MethodSpecification(
    config=TrainerConfig(
        method_name="seathru-nerf-lite",
        steps_per_eval_batch=500,
        steps_per_save=2000,
        max_num_iterations=50000,
        mixed_precision=True,
        pipeline=VanillaPipelineConfig(
            datamanager=VanillaDataManagerConfig(
                dataparser=NerfstudioDataParserConfig(),
                train_num_rays_per_batch=8192,
                eval_num_rays_per_batch=4096,
            ),
            model=SeathruModelConfig(
                eval_num_rays_per_chunk=1 << 15,
                num_nerf_samples_per_ray=64,
                num_proposal_samples_per_ray=(256, 128),
                max_res=2048,
                log2_hashmap_size=19,
                hidden_dim=64,
                bottleneck_dim=31,
                hidden_dim_colour=64,
                hidden_dim_medium=64,
                proposal_net_args_list=[
                    {
                        "hidden_dim": 16,
                        "log2_hashmap_size": 17,
                        "num_levels": 5,
                        "max_res": 128,
                        "use_linear": False,
                    },
                    {
                        "hidden_dim": 16,
                        "log2_hashmap_size": 17,
                        "num_levels": 5,
                        "max_res": 256,
                        "use_linear": False,
                    },
                ],
            ),
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=2e-3, eps=1e-8),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1e-5, max_steps=500000, warmup_steps=1024
                ),
            },
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=2e-3, eps=1e-8, max_norm=0.001),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1e-5, max_steps=500000, warmup_steps=1024
                ),
            },
            "camera_opt": {
                "mode": "off",
                "optimizer": AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=6e-6, max_steps=500000
                ),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="Light SeaThru-NeRF for underwater scenes.",
)

"seathru_field.py"
import torch
from torch import nn
from torch import Tensor

from typing import Dict, Literal, Optional, Tuple
from nerfstudio.cameras.rays import RaySamples
from nerfstudio.fields.base_field import Field, get_normalized_directions
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.mlp import MLP
from nerfstudio.field_components.activations import trunc_exp
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.field_components.encodings import HashEncoding, SHEncoding
from nerfstudio.field_components.spatial_distortions import SpatialDistortion

from seathru.seathru_fieldheadnames import SeathruHeadNames

try:
    import tinycudann as tcnn  # noqa
except ImportError:
    print("tinycudann is not installed! Please install it for faster training.")


class SeathruField(Field):
    """Field for Seathru-NeRF. Default configuration is the big model.

    Args:
        aabb: parameters of scene aabb bounds
        num_levels: number of levels of the hashmap for the object base MLP
        min_res: minimum resolution of the hashmap for the object base MLP
        max_res: maximum resolution of the hashmap for the object base MLP
        log2_hashmap_size: size of the hashmap for the object base MLP
        features_per_level: number of features per level of the hashmap for the object
                            base MLP
        num_layers: number of hidden layers for the object base MLP
        hidden_dim: dimension of hidden layers for the object base MLP
        bottleneck_dim: bottleneck dimension between object base MLP and object head MLP
        num_layers_colour: number of hidden layers for colour MLP
        hidden_dim_colour: dimension of hidden layers for colour MLP
        num_layers_medium: number of hidden layers for medium MLP
        hidden_dim_medium: dimension of hidden layers for medium MLP
        spatial_distortion: spatial distortion to apply to the scene
        implementation: implementation of the base mlp (tcnn or torch)
        use_viewing_dir_obj_rgb: whether to use viewing direction in object rgb MLP
        object_density_bias: bias for object density
        medium_density_bias: bias for medium density
    """

    aabb: Tensor

    def __init__(
        self,
        aabb: Tensor,
        num_levels: int = 16,
        min_res: int = 16,
        max_res: int = 8192,
        log2_hashmap_size: int = 21,
        features_per_level: int = 2,
        num_layers: int = 2,
        hidden_dim: int = 256,
        bottleneck_dim: int = 63,
        num_layers_colour: int = 3,
        hidden_dim_colour: int = 256,
        num_layers_medium: int = 2,
        hidden_dim_medium: int = 128,
        spatial_distortion: Optional[SpatialDistortion] = None,
        implementation: Literal["tcnn", "torch"] = "tcnn",
        use_viewing_dir_obj_rgb: bool = False,
        object_density_bias: float = 0.0,
        medium_density_bias: float = 0.0,
    ) -> None:
        super().__init__()

        # Register buffers
        self.register_buffer("aabb", aabb)
        self.register_buffer("max_res", torch.tensor(max_res))
        self.register_buffer("num_levels", torch.tensor(num_levels))
        self.register_buffer("log2_hashmap_size", torch.tensor(log2_hashmap_size))

        self.bottleneck_dim = bottleneck_dim
        self.spatial_distortion = spatial_distortion
        self.use_viewing_dir_obj_rgb = use_viewing_dir_obj_rgb
        self.object_density_bias = object_density_bias
        self.medium_density_bias = medium_density_bias
        self.direction_encoding = SHEncoding(levels=4, implementation=implementation)
        self.colour_activation = nn.Sigmoid()
        self.sigma_activation = nn.Softplus()

        # ------------------------Object network------------------------
        # Position encoding with trainable hash map
        self.hash_map = HashEncoding(
            num_levels=num_levels,
            min_res=min_res,
            max_res=max_res,
            log2_hashmap_size=log2_hashmap_size,
            features_per_level=features_per_level,
            implementation=implementation,
        )

        # Slim mlp for object
        self.object_mlp_base_mlp = MLP(
            in_dim=self.hash_map.get_out_dim(),
            num_layers=num_layers,
            layer_width=hidden_dim,
            out_dim=1 + self.bottleneck_dim,
            activation=nn.ReLU(),
            out_activation=None,
            implementation=implementation,
        )

        # Object mlp_base
        self.object_mlp_base = torch.nn.Sequential(
            self.hash_map, self.object_mlp_base_mlp
        )

        # Object colour MLP
        direction_enc_out_dim = 0
        if self.use_viewing_dir_obj_rgb:
            direction_enc_out_dim = self.direction_encoding.get_out_dim()

        self.mlp_colour = MLP(
            in_dim=direction_enc_out_dim + self.bottleneck_dim,
            num_layers=num_layers_colour,
            layer_width=hidden_dim_colour,
            out_dim=3,
            activation=nn.ReLU(),
            out_activation=nn.Sigmoid(),
            implementation=implementation,
        )

        # ------------------------Medium network------------------------
        # Medium MLP
        self.medium_mlp = MLP(
            in_dim=self.direction_encoding.get_out_dim(),
            num_layers=num_layers_medium,
            layer_width=hidden_dim_medium,
            out_dim=9,
            activation=nn.Softplus(),
            out_activation=None,
            implementation=implementation,
        )

    def get_density(self, ray_samples: RaySamples) -> Tuple[Tensor, Tensor]:
        """Compute output of object base MLP. (This function builds on the nerfacto
           implementation)

        Args:
            ray_samples: RaySamples object containing the ray samples.

        Returns:
            Tuple containing the object density and the bottleneck vector.
        """
        if self.spatial_distortion is not None:
            positions = ray_samples.frustums.get_positions()
            positions = self.spatial_distortion(positions)
            # Normalize positions from [-2, 2] to [0, 1]
            positions = (positions + 2.0) / 4.0
        else:
            # If working with scene box instead of scene contraction
            positions = SceneBox.get_normalized_positions(
                ray_samples.frustums.get_positions(), self.aabb
            )

        # Make sure the tcnn gets inputs between 0 and 1
        selector = ((positions > 0.0) & (positions < 1.0)).all(dim=-1)
        positions = positions * selector[..., None]
        self._sample_locations = positions

        # Make sure to turn gradients on for the sample locations
        if not self._sample_locations.requires_grad:
            self._sample_locations.requires_grad = True

        # Forward pass through the object base MLP
        positions_flat = positions.view(-1, 3)
        h_object = self.object_mlp_base(positions_flat).view(
            *ray_samples.frustums.shape, -1
        )
        density_before_activation, bottleneck_vector = torch.split(
            h_object, [1, self.bottleneck_dim], dim=-1
        )

        # From nerfacto: "Rectifying the density with an exponential is much more stable
        # than a ReLU or softplus, because it enables high post-activation (float32)
        # density outputs from smaller internal (float16) parameters."
        density_before_activation = density_before_activation + self.object_density_bias
        density = trunc_exp(density_before_activation.to(positions))
        density = density * selector[..., None]
        return density, bottleneck_vector

    def get_outputs(  # type: ignore
        self, ray_samples: RaySamples, density_embedding: Optional[Tensor] = None
    ) -> Dict[SeathruHeadNames, Tensor]:
        """Compute outputs of object and medium networks (except object density).

        Args:
            ray_samples: RaySamples object containing the ray samples.
            density_embedding: Bottleneck vector (output of object base MLP).

        Returns:
            Dictionary containing the outputs seathru network.
        """
        assert density_embedding is not None
        outputs = {}

        # Encode directions
        directions = get_normalized_directions(ray_samples.frustums.directions)
        directions_flat = directions.view(-1, 3)
        directions_encoded = self.direction_encoding(directions_flat)

        outputs_shape = ray_samples.frustums.directions.shape[:-1]

        if self.use_viewing_dir_obj_rgb:
            h_object = torch.cat(
                [directions_encoded, density_embedding.view(-1, self.bottleneck_dim)],
                dim=-1,
            )
        else:
            h_object = density_embedding.view(-1, self.bottleneck_dim)

        # Object colour MLP forward pass
        rgb_object = self.mlp_colour(h_object).view(*outputs_shape, -1).to(directions)
        outputs[FieldHeadNames.RGB] = rgb_object

        # Medium MLP forward pass
        medium_base_out = self.medium_mlp(directions_encoded)

        # different activations for different outputs
        medium_rgb = (
            self.colour_activation(medium_base_out[..., :3])
            .view(*outputs_shape, -1)
            .to(directions)
        )
        medium_bs = (
            self.sigma_activation(medium_base_out[..., 3:6] + self.medium_density_bias)
            .view(*outputs_shape, -1)
            .to(directions)
        )
        medium_attn = (
            self.sigma_activation(medium_base_out[..., 6:] + self.medium_density_bias)
            .view(*outputs_shape, -1)
            .to(directions)
        )

        outputs[SeathruHeadNames.MEDIUM_RGB] = medium_rgb
        outputs[SeathruHeadNames.MEDIUM_BS] = medium_bs
        outputs[SeathruHeadNames.MEDIUM_ATTN] = medium_attn

        return outputs
    


"seathru_losses.py"

import torch
from torch import Tensor
from jaxtyping import Float


def acc_loss(
    transmittance_object: Float[Tensor, "*bs num_samples 1"], beta: float
) -> torch.Tensor:
    """Compute the acc_loss.

    Args:
        transmittance_object: Transmittances of object.
        factor: factor to control the weight of the two distributions

    Returns:
            Acc Loss.
    """
    P = torch.exp(-torch.abs(transmittance_object) / 0.1) + beta * torch.exp(
        -torch.abs(1 - transmittance_object) / 0.1
    )
    loss = -torch.log(P)
    return loss.mean()


def recon_loss(gt: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    """Compute the reconstruction loss.

    Args:
        gt: Ground truth.
        pred: RGB prediction.

    Returns:
        Reconstruction loss.
    """
    inner = torch.square((pred - gt) / (pred.detach() + 1e-3))
    return torch.mean(inner)

"seathru_model.py"

from dataclasses import dataclass, field
from typing import Dict, List, Type, Literal, Tuple

import numpy as np
import torch
from torch.nn import Parameter
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.cameras.rays import RayBundle, RaySamples
from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.model_components.scene_colliders import NearFarCollider
from nerfstudio.model_components.renderers import AccumulationRenderer
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.model_components.losses import MSELoss, interlevel_loss
from nerfstudio.utils import colormaps
from nerfstudio.fields.density_fields import HashMLPDensityField
from nerfstudio.model_components.ray_samplers import (
    ProposalNetworkSampler,
    UniformSampler,
)


from seathru.seathru_field import SeathruField
from seathru.seathru_fieldheadnames import SeathruHeadNames
from seathru.seathru_renderers import SeathruRGBRenderer
from seathru.seathru_losses import acc_loss, recon_loss
from seathru.seathru_utils import get_bayer_mask, save_debug_info, get_transmittance
from seathru.seathru_renderers import SeathruDepthRenderer


@dataclass
class SeathruModelConfig(ModelConfig):
    """SeaThru-NeRF Config."""

    _target: Type = field(default_factory=lambda: SeathruModel)
    near_plane: float = 0.05
    """Near plane of rays."""
    far_plane: float = 10.0
    """Far plane of rays."""
    num_levels: int = 16
    """Number of levels of the hashmap for the object base MLP."""
    min_res: int = 16
    """Minimum resolution of the hashmap for the object base MLP."""
    max_res: int = 8192
    """Maximum resolution of the hashmap for the object base MLP."""
    log2_hashmap_size: int = 21
    """Size of the hashmap for the object base MLP."""
    features_per_level: int = 2
    """Number of features per level of the hashmap for the object base MLP."""
    num_layers: int = 2
    """Number of hidden layers for the object base MLP."""
    hidden_dim: int = 256
    """Dimension of hidden layers for the object base MLP."""
    bottleneck_dim: int = 63
    """Bottleneck dimension between object base MLP and object head MLP."""
    num_layers_colour: int = 3
    """Number of hidden layers for colour MLP."""
    hidden_dim_colour: int = 256
    """Dimension of hidden layers for colour MLP."""
    num_layers_medium: int = 2
    """Number of hidden layers for medium MLP."""
    hidden_dim_medium: int = 128
    """Dimension of hidden layers for medium MLP."""
    implementation: Literal["tcnn", "torch"] = "tcnn"
    """Implementation of the MLPs (tcnn or torch)."""
    use_viewing_dir_obj_rgb: bool = False
    """Whether to use viewing direction in object rgb MLP."""
    object_density_bias: float = 0.0
    """Bias for object density."""
    medium_density_bias: float = 0.0
    """Bias for medium density (sigma_bs and sigma_attn)."""
    num_proposal_samples_per_ray: Tuple[int, ...] = (256, 128)
    """Number of samples per ray for each proposal network."""
    num_nerf_samples_per_ray: int = 64
    """Number of samples per ray for the nerf network."""
    proposal_update_every: int = 5
    """Sample every n steps after the warmup."""
    proposal_warmup: int = 5000
    """Scales n from 1 to proposal_update_every over this many steps."""
    num_proposal_iterations: int = 2
    """Number of proposal network iterations."""
    use_same_proposal_network: bool = False
    """Whether to use the same proposal network."""
    proposal_net_args_list: List[Dict] = field(
        default_factory=lambda: [
            {
                "hidden_dim": 16,
                "log2_hashmap_size": 17,
                "num_levels": 5,
                "max_res": 512,
                "use_linear": False,
            },
            {
                "hidden_dim": 16,
                "log2_hashmap_size": 17,
                "num_levels": 7,
                "max_res": 2048,
                "use_linear": False,
            },
        ]
    )
    """Arguments for the proposal density fields."""
    proposal_initial_sampler: Literal["piecewise", "uniform"] = "piecewise"
    """Initial sampler for the proposal network."""
    interlevel_loss_mult: float = 1.0
    """Proposal loss multiplier."""
    use_proposal_weight_anneal: bool = True
    """Whether to use proposal weight annealing (this gives an exploration at the \
        beginning of training)."""
    proposal_weights_anneal_slope: float = 10.0
    """Slope of the annealing function for the proposal weights."""
    proposal_weights_anneal_max_num_iters: int = 15000
    """Max num iterations for the annealing function."""
    use_single_jitter: bool = True
    """Whether use single jitter or not for the proposal networks."""
    disable_scene_contraction: bool = False
    """Whether to disable scene contraction or not."""
    use_gradient_scaling: bool = False
    """Use gradient scaler where the gradients are lower for points closer to \
        the camera."""
    initial_acc_loss_mult: float = 0.0001
    """Initial accuracy loss multiplier."""
    final_acc_loss_mult: float = 0.0001
    """Final accuracy loss multiplier."""
    acc_decay: int = 10000
    """Decay of the accuracy loss multiplier. (After this many steps, acc_loss_mult = \
        final_acc_loss_mult.)"""
    rgb_loss_use_bayer_mask: bool = False
    """Whether to use a Bayer mask for the RGB loss."""
    prior_on: Literal["weights", "transmittance"] = "transmittance"
    """Prior on the proposal weights or transmittance."""
    debug: bool = False
    """Whether to save debug information."""
    beta_prior: float = 100.0
    """Beta hyperparameter for the prior used in the acc_loss."""
    use_viewing_dir_obj_rgb: bool = False
    """Whether to use viewing direction in object rgb MLP."""
    use_new_rendering_eqs: bool = True
    """Whether to use the new rendering equations."""


class SeathruModel(Model):
    """Seathru model

    Args:
        config: SeaThru-NeRF configuration to instantiate the model with.
    """

    config: SeathruModelConfig  # type: ignore

    def populate_modules(self):
        """Setup the fields and modules."""
        super().populate_modules()

        # Scene contraction
        if self.config.disable_scene_contraction:
            scene_contraction = None
        else:
            scene_contraction = SceneContraction(order=float("inf"))

        # Initialize SeaThru field
        self.field = SeathruField(
            aabb=self.scene_box.aabb,
            num_levels=self.config.num_levels,
            min_res=self.config.min_res,
            max_res=self.config.max_res,
            log2_hashmap_size=self.config.log2_hashmap_size,
            features_per_level=self.config.features_per_level,
            num_layers=self.config.num_layers,
            hidden_dim=self.config.hidden_dim,
            bottleneck_dim=self.config.bottleneck_dim,
            num_layers_colour=self.config.num_layers_colour,
            hidden_dim_colour=self.config.hidden_dim_colour,
            num_layers_medium=self.config.num_layers_medium,
            hidden_dim_medium=self.config.hidden_dim_medium,
            spatial_distortion=scene_contraction,
            implementation=self.config.implementation,
            use_viewing_dir_obj_rgb=self.config.use_viewing_dir_obj_rgb,
            object_density_bias=self.config.object_density_bias,
            medium_density_bias=self.config.medium_density_bias,
        )

        # Initialize proposal network(s) (this code snippet is taken from from nerfacto)
        self.density_fns = []
        num_prop_nets = self.config.num_proposal_iterations
        # Build the proposal network(s)
        self.proposal_networks = torch.nn.ModuleList()
        if self.config.use_same_proposal_network:
            assert (
                len(self.config.proposal_net_args_list) == 1
            ), "Only one proposal network is allowed."
            prop_net_args = self.config.proposal_net_args_list[0]
            network = HashMLPDensityField(
                self.scene_box.aabb,
                spatial_distortion=scene_contraction,
                **prop_net_args,
                implementation=self.config.implementation,
            )
            self.proposal_networks.append(network)
            self.density_fns.extend([network.density_fn for _ in range(num_prop_nets)])
        else:
            for i in range(num_prop_nets):
                prop_net_args = self.config.proposal_net_args_list[
                    min(i, len(self.config.proposal_net_args_list) - 1)
                ]
                network = HashMLPDensityField(
                    self.scene_box.aabb,
                    spatial_distortion=scene_contraction,
                    **prop_net_args,
                    implementation=self.config.implementation,
                )
                self.proposal_networks.append(network)
            self.density_fns.extend(
                [network.density_fn for network in self.proposal_networks]
            )

        def update_schedule(step):
            return np.clip(
                np.interp(
                    step,
                    [0, self.config.proposal_warmup],
                    [0, self.config.proposal_update_every],
                ),
                1,
                self.config.proposal_update_every,
            )

        # Initial sampler
        initial_sampler = None  # None is for piecewise as default
        if self.config.proposal_initial_sampler == "uniform":
            initial_sampler = UniformSampler(
                single_jitter=self.config.use_single_jitter
            )

        # Proposal sampler
        self.proposal_sampler = ProposalNetworkSampler(
            num_nerf_samples_per_ray=self.config.num_nerf_samples_per_ray,
            num_proposal_samples_per_ray=self.config.num_proposal_samples_per_ray,
            num_proposal_network_iterations=self.config.num_proposal_iterations,
            single_jitter=self.config.use_single_jitter,
            update_sched=update_schedule,
            initial_sampler=initial_sampler,
        )

        # Collider
        self.collider = NearFarCollider(
            near_plane=self.config.near_plane, far_plane=self.config.far_plane
        )

        # Renderers
        self.renderer_rgb = SeathruRGBRenderer(
            use_new_rendering_eqs=self.config.use_new_rendering_eqs
        )
        self.renderer_depth = SeathruDepthRenderer(
            far_plane=self.config.far_plane, method="median"
        )
        self.renderer_accumulation = AccumulationRenderer()

        # Losses
        self.rgb_loss = MSELoss(reduction="none")

        # Metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)

        # Step member variable to keep track of the training step
        self.step = 0

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Get the parameter groups for the optimizer. (Code snippet from nerfacto)

        Returns:
            The parameter groups.
        """
        param_groups = {}
        param_groups["proposal_networks"] = list(self.proposal_networks.parameters())
        param_groups["fields"] = list(self.field.parameters())
        return param_groups

    def step_cb(self, step) -> None:
        """Function for training callbacks to use to update training step.

        Args:
            step: The training step.
        """
        self.step = step

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        """Get the training callbacks.
           (Code of this function is from nerfacto but added step tracking for debugging.)

        Args:
            training_callback_attributes: The training callback attributes.

        Returns:
            List with training callbacks.
        """
        callbacks = []
        if self.config.use_proposal_weight_anneal:
            # anneal the weights of the proposal network before doing PDF sampling
            N = self.config.proposal_weights_anneal_max_num_iters

            def set_anneal(step):
                # https://arxiv.org/pdf/2111.12077.pdf eq. 18
                train_frac = np.clip(step / N, 0, 1)

                def bias(x, b):
                    return b * x / ((b - 1) * x + 1)

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

        # Additional callback to track the training step for decaying and
        # debugging purposes
        callbacks.append(
            TrainingCallback(
                where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                update_every_num_iters=1,
                func=self.step_cb,
            )
        )

        return callbacks

    def get_outputs(self, ray_bundle: RayBundle) -> Dict[str, torch.Tensor]:  # type: ignore
        """Get outputs from the model.

        Args:
            ray_bundle: RayBundle containing the input rays to compute and render.

        Returns:
            Dict containing the outputs of the model.
        """

        ray_samples: RaySamples

        # Get output from proposal network(s)
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler(
            ray_bundle, density_fns=self.density_fns
        )

        # Get output from Seathru field
        field_outputs = self.field.forward(ray_samples)
        field_outputs[FieldHeadNames.DENSITY] = torch.nan_to_num(
            field_outputs[FieldHeadNames.DENSITY], nan=1e-3
        )
        weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])
        weights_list.append(weights)
        ray_samples_list.append(ray_samples)

        # Render rgb (only rgb in training and rgb, direct, bs, J in eval for
        # performance reasons as we do not optimize with respect to direct, bs, J)
        # ignore types to avoid unnecesarry pyright errors
        if self.training or not self.config.use_new_rendering_eqs:
            rgb = self.renderer_rgb(
                object_rgb=field_outputs[FieldHeadNames.RGB],
                medium_rgb=field_outputs[SeathruHeadNames.MEDIUM_RGB],  # type: ignore
                medium_bs=field_outputs[SeathruHeadNames.MEDIUM_BS],  # type: ignore
                medium_attn=field_outputs[SeathruHeadNames.MEDIUM_ATTN],  # type: ignore
                densities=field_outputs[FieldHeadNames.DENSITY],
                weights=weights,
                ray_samples=ray_samples,
            )
            direct = None
            bs = None
            J = None
        else:
            rgb, direct, bs, J = self.renderer_rgb(
                object_rgb=field_outputs[FieldHeadNames.RGB],
                medium_rgb=field_outputs[SeathruHeadNames.MEDIUM_RGB],  # type: ignore
                medium_bs=field_outputs[SeathruHeadNames.MEDIUM_BS],  # type: ignore
                medium_attn=field_outputs[SeathruHeadNames.MEDIUM_ATTN],  # type: ignore
                densities=field_outputs[FieldHeadNames.DENSITY],
                weights=weights,
                ray_samples=ray_samples,
            )

        # Render depth and accumulation
        depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
        accumulation = self.renderer_accumulation(weights=weights)

        # Calculate transmittance and add to outputs for acc_loss calculation
        # Ignore type error that occurs because ray_samples can be initialized without deltas
        transmittance = get_transmittance(
            ray_samples.deltas, field_outputs[FieldHeadNames.DENSITY]  # type: ignore
        )
        outputs = {
            "rgb": rgb,
            "depth": depth,
            "accumulation": accumulation,
            "transmittance": transmittance,
            "weights": weights,
            "direct": direct if not self.training else None,
            "bs": bs if not self.training else None,
            "J": J if not self.training else None,
        }

        # Add outputs from proposal network(s) to outputs if training for proposal loss
        if self.training:
            outputs["weights_list"] = weights_list
            outputs["ray_samples_list"] = ray_samples_list

        # Add proposed depth to outputs
        for i in range(self.config.num_proposal_iterations):
            outputs[f"prop_depth_{i}"] = self.renderer_depth(
                weights=weights_list[i], ray_samples=ray_samples_list[i]
            )

        return outputs

    def get_metrics_dict(self, outputs, batch):
        """Get evaluation metrics dictionary.
        (Compared to get_image_metrics_and_images(), this function does not render
        images and is executed at each training step.)

        Args:
            outputs: Dict containing the outputs of the model.
            batch: Dict containing the gt data.

        Returns:
            Dict containing the metrics to log.
        """
        metrics_dict = {}
        gt_rgb = batch["image"].to(self.device)
        predicted_rgb = outputs["rgb"]
        metrics_dict["psnr"] = self.psnr(predicted_rgb, gt_rgb)
        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        """Calculate loss dictionary.

        Args:
            outputs: Dict containing the outputs of the model.
            batch: Dict containing the gt data.

        Returns:
            Dict containing the loss values.
        """
        loss_dict = {}
        image = batch["image"].to(self.device)

        # RGB loss
        if self.config.rgb_loss_use_bayer_mask:
            # Cut out camera/image indices and pass to get_bayer_mask
            bayer_mask = get_bayer_mask(batch["indices"][:, 1:].to(self.device))
            squared_error = self.rgb_loss(image, outputs["rgb"])  # clip or not clip?
            scaling_grad = 1 / (outputs["rgb"].detach() + 1e-3)
            loss = squared_error * torch.square(scaling_grad)
            denom = torch.sum(bayer_mask)
            loss_dict["rgb_loss"] = torch.sum(loss * bayer_mask) / denom
        else:
            loss_dict["rgb_loss"] = recon_loss(gt=image, pred=outputs["rgb"])

        if self.training:
            # Accumulation loss
            if self.step < self.config.acc_decay:
                acc_loss_mult = self.config.initial_acc_loss_mult
            else:
                acc_loss_mult = self.config.final_acc_loss_mult

            if self.config.prior_on == "weights":
                loss_dict["acc_loss"] = acc_loss_mult * acc_loss(
                    transmittance_object=outputs["weights"], beta=self.config.beta_prior
                )
            elif self.config.prior_on == "transmittance":
                loss_dict["acc_loss"] = acc_loss_mult * acc_loss(
                    transmittance_object=outputs["transmittance"],
                    beta=self.config.beta_prior,
                )
            else:
                raise ValueError(f"Unknown prior_on: {self.config.prior_on}")

            # Proposal loss
            loss_dict["interlevel_loss"] = (
                self.config.interlevel_loss_mult
                * interlevel_loss(outputs["weights_list"], outputs["ray_samples_list"])
            )
        return loss_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        """Get evaluation metrics dictionary and images to log for eval batch.
        (extended from nerfacto)

        Args:
            outputs: Dict containing the outputs of the model.
            batch: Dict containing the gt data.

        Returns:
            Tuple containing the metrics to log (as scalars) and the images to log.
        """
        image = batch["image"].to(self.device)
        rgb = outputs["rgb"]

        # Accumulation and depth maps
        acc = colormaps.apply_colormap(outputs["accumulation"])
        depth = colormaps.apply_depth_colormap(outputs["depth"])

        combined_rgb = torch.cat([image, rgb], dim=1)
        combined_acc = torch.cat([acc], dim=1)
        combined_depth = torch.cat([depth], dim=1)

        # Log the images
        images_dict = {
            "img": combined_rgb,
            "accumulation": combined_acc,
            "depth": combined_depth,
        }

        if self.config.use_new_rendering_eqs:
            # J (clean image), direct and bs images
            direct = outputs["direct"]
            bs = outputs["bs"]
            J = outputs["J"]

            combined_direct = torch.cat([direct], dim=1)
            combined_bs = torch.cat([bs], dim=1)
            combined_J = torch.cat([J], dim=1)

            # log the images
            images_dict["direct"] = combined_direct
            images_dict["bs"] = combined_bs
            images_dict["J"] = combined_J

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        image = torch.moveaxis(image, -1, 0)[None, ...]
        rgb = torch.moveaxis(rgb, -1, 0)[None, ...]

        # Compute metrics
        psnr = self.psnr(image, rgb)
        ssim = self.ssim(image, rgb)
        lpips = self.lpips(image, rgb)

        # Log the metrics (as scalars)
        metrics_dict = {"psnr": float(psnr.item()), "ssim": float(ssim)}  # type: ignore
        metrics_dict["lpips"] = float(lpips)

        # Log the proposal depth maps
        for i in range(self.config.num_proposal_iterations):
            key = f"prop_depth_{i}"
            prop_depth_i = colormaps.apply_depth_colormap(outputs[key])
            images_dict[key] = prop_depth_i

        # Debugging
        if self.config.debug:
            save_debug_info(
                weights=outputs["weights"],
                transmittance=outputs["transmittance"],
                depth=outputs["depth"],
                prop_depth=outputs["prop_depth_0"],
                step=self.step,
            )

        return metrics_dict, images_dict
    
"seathru_renderers.py"
from typing import Literal, Union, Tuple

import torch
from jaxtyping import Float
from torch import Tensor, nn

from nerfstudio.cameras.rays import RaySamples

from seathru.seathru_utils import get_transmittance


class SeathruRGBRenderer(nn.Module):
    """Volumetric RGB rendering of an unnderwater scene.

    Args:
        use_new_rendering_eqs: Whether to use the new rendering equations.
    """

    def __init__(self, use_new_rendering_eqs: bool = True) -> None:
        super().__init__()
        self.use_new_rendering_eqs = use_new_rendering_eqs

    def combine_rgb(
        self,
        object_rgb: Float[Tensor, "*bs num_samples 3"],
        medium_rgb: Float[Tensor, "*bs num_samples 3"],
        medium_bs: Float[Tensor, "*bs num_samples 3"],
        medium_attn: Float[Tensor, "*bs num_samples 3"],
        densities: Float[Tensor, "*bs num_samples 1"],
        weights: Float[Tensor, "*bs num_samples 1"],
        ray_samples: RaySamples,
    ) -> Union[Tensor, Tuple[Tensor, Tensor, Tensor, Tensor]]:
        """Render pixel colour along rays using volumetric rendering.

        Args:
            object_rgb: RGB values of object.
            medium_rgb: RGB values of medium.
            medium_bs:  sigma backscatter of medium.
            medium_attn: sigma attenuation of medium.
            densities: Object densities.
            weights: Object weights.
            ray_samples: Set of ray samples.

        Returns:
            Rendered pixel colour (and direct, bs, and J if using new rendering
            equations and not training).
        """

        # Old rendering equations
        if not self.use_new_rendering_eqs:
            s = ray_samples.frustums.starts

            # Object RGB
            attn_component = torch.exp(-medium_attn * s)
            comp_object_rgb = torch.sum(weights * attn_component * object_rgb, dim=-2)

            # Medium RGB
            # Ignore type error that occurs because ray_samples can be initialized without deltas
            transmittance_object = get_transmittance(ray_samples.deltas, densities)  # type: ignore
            bs_comp1 = torch.exp(-medium_bs * s)
            bs_comp2 = 1 - torch.exp(-medium_bs * ray_samples.deltas)
            comp_medium_rgb = torch.sum(
                transmittance_object * bs_comp1 * bs_comp2 * medium_rgb, dim=-2
            )
            comp_medium_rgb = torch.nan_to_num(comp_medium_rgb)

            comp_rgb = comp_object_rgb + comp_medium_rgb

            return comp_rgb

        # New rendering equations (adapted from https://github.com/deborahLevy130/seathru_NeRF/blob/c195ff3384632058d56aef0cddb8057b538c1511/internal/render.py#L288C8-L288C8)
        # Lead to the same comp_rgb as the old rendering equations, but also return
        # direct, bs, and J. (and detach deltas for medium contributions as it showed
        # to enhance stability)
        else:
            # Ignore type error that occurs because ray_samples can be initialized without deltas
            transmittance_object = get_transmittance(ray_samples.deltas, densities)  # type: ignore

            # Get transmittance_attn
            # Ignore type error that occurs because ray_samples can be initialized without deltas
            deltas_detached = ray_samples.deltas.detach()  # type: ignore
            transmittance_attn = get_transmittance(
                deltas=deltas_detached, densities=medium_attn
            )

            # Get bs_weights
            transmittance_bs = get_transmittance(
                deltas=deltas_detached, densities=medium_bs
            )
            alphas_bs = 1 - torch.exp(-medium_bs * deltas_detached)
            bs_weights = alphas_bs * transmittance_bs

            # Get direct and bs
            direct = torch.sum(weights * transmittance_attn * object_rgb, dim=-2)
            bs = torch.sum(transmittance_object * bs_weights * medium_rgb, dim=-2)
            comp_rgb = direct + bs
            J = (torch.sum(weights * object_rgb, dim=-2)).detach()

            # Only return direct, bs, and J if not training to save memory and time
            if not self.training:
                return comp_rgb, direct, bs, J
            else:
                return comp_rgb

    def forward(
        self,
        object_rgb: Float[Tensor, "*bs num_samples 3"],
        medium_rgb: Float[Tensor, "*bs num_samples 3"],
        medium_bs: Float[Tensor, "*bs num_samples 3"],
        medium_attn: Float[Tensor, "*bs num_samples 3"],
        densities: Float[Tensor, "*bs num_samples 1"],
        weights: Float[Tensor, "*bs num_samples 1"],
        ray_samples: RaySamples,
    ) -> Union[Tensor, Tuple[Tensor, Tensor, Tensor, Tensor]]:
        """Render pixel colour along rays using volumetric rendering.

        Args:
            object_rgb: RGB values of object.
            medium_rgb: RGB values of medium.
            medium_bs:  sigma backscatter of medium.
            medium_attn: sigma attenuation of medium.
            densities: Object densities.
            weights: Object weights.
            ray_samples: Set of ray samples.

        Returns:
            Rendered pixel colour (and direct, bs, and J if using new rendering
            equations and not training).
        """

        if not self.training:
            object_rgb = torch.nan_to_num(object_rgb)
            medium_rgb = torch.nan_to_num(medium_rgb)
            medium_bs = torch.nan_to_num(medium_bs)
            medium_attn = torch.nan_to_num(medium_attn)

            if self.use_new_rendering_eqs:
                rgb, direct, bs, J = self.combine_rgb(
                    object_rgb,
                    medium_rgb,
                    medium_bs,
                    medium_attn,
                    densities,
                    weights,
                    ray_samples=ray_samples,
                )

                torch.clamp_(rgb, min=0.0, max=1.0)
                torch.clamp_(direct, min=0.0, max=1.0)
                torch.clamp_(bs, min=0.0, max=1.0)
                torch.clamp_(J, min=0.0, max=1.0)

                return rgb, direct, bs, J

            else:
                rgb = self.combine_rgb(
                    object_rgb,
                    medium_rgb,
                    medium_bs,
                    medium_attn,
                    densities,
                    weights,
                    ray_samples=ray_samples,
                )

                if isinstance(rgb, torch.Tensor):
                    torch.clamp_(rgb, min=0.0, max=1.0)

                return rgb

        else:
            rgb = self.combine_rgb(
                object_rgb,
                medium_rgb,
                medium_bs,
                medium_attn,
                densities,
                weights,
                ray_samples=ray_samples,
            )
            return rgb


class SeathruDepthRenderer(nn.Module):
    """Calculate depth along rays.

    Args:
        far_plane: Far plane of rays.
        method: Depth calculation method.
    """

    def __init__(
        self, far_plane: float, method: Literal["median", "expected"] = "median"
    ) -> None:
        super().__init__()
        self.far_plane = far_plane
        self.method = method

    def forward(
        self,
        weights: Float[Tensor, "*batch num_samples 1"],
        ray_samples: RaySamples,
    ) -> Float[Tensor, "*batch 1"]:
        """Calculate depth along rays.

        Args:
            weights: Weights for each sample.
            ray_samples: Set of ray samples.

        Returns:
            Depht values.
        """
        steps = (ray_samples.frustums.starts + ray_samples.frustums.ends) / 2
        acc_weights = torch.sum(
            weights, dim=-2, keepdim=True
        )  # Shape: [num_rays, 1, 1]

        # As object weights are not guaranteed to sum to 1 (e.g. when in a region of
        # only water without an object), we need to add an additional sample at the end
        # of each ray to ensure that the weights sum to 1.

        # Compute the weight for the additional sample
        bg_weight = 1.0 - acc_weights

        # Concatenate this new weight to the original weights tensor
        weights_ext = torch.cat([weights, bg_weight], dim=1)
        # Concatenate the far plane to the original steps tensor
        steps_ext = torch.cat(
            [
                steps,
                torch.ones((*steps.shape[:1], 1, 1), device=steps.device)
                * self.far_plane,
            ],
            dim=1,
        )

        if self.method == "expected":
            eps = 1e-10
            depth = torch.sum(weights * steps, dim=-2) / (torch.sum(weights, -2) + eps)
            depth = torch.clip(depth, steps.min(), steps.max())
            return depth

        if self.method == "median":
            # Code snippet from nerfstudio DepthRenderer
            cumulative_weights_ext = torch.cumsum(
                weights_ext[..., 0], dim=-1
            )  # [..., num_samples]
            split = (
                torch.ones((*weights_ext.shape[:-2], 1), device=weights_ext.device)
                * 0.5
            )  # [..., 1]
            median_index = torch.searchsorted(
                cumulative_weights_ext, split, side="left"
            )  # [..., 1]
            median_index = torch.clamp(
                median_index, 0, steps_ext.shape[-2] - 1
            )  # [..., 1]
            median_depth = torch.gather(
                steps_ext[..., 0], dim=-1, index=median_index
            )  # [..., 1]
            return median_depth

        raise NotImplementedError(f"Method {self.method} not implemented")


"seathru_utils.py"

from jaxtyping import Float

import torch
import os
from torch import Tensor


def get_transmittance(
    deltas: Tensor, densities: Float[Tensor, "*bs num_samples 1"]
) -> Float[Tensor, "*bs num_samples 1"]:
    """Compute transmittance for each ray sample.

    Args:
        deltas: Distance between each ray sample.
        densities: Densities of each ray sample.

    Returns:
        Transmittance for each ray sample.
    """
    delta_density = deltas * densities
    transmittance_object = torch.cumsum(delta_density[..., :-1, :], dim=-2)
    transmittance_object = torch.cat(
        [
            torch.zeros(
                (*transmittance_object.shape[:1], 1, transmittance_object.shape[-1]),
                device=transmittance_object.device,
            ),
            transmittance_object,
        ],
        dim=-2,
    )
    transmittance_object = torch.exp(-transmittance_object)

    return transmittance_object


def get_bayer_mask(indices: torch.Tensor) -> torch.Tensor:
    """Get bayer mask for rgb loss.

    Args:
        indices: tensor of shape ([num_rays, 2]) containing the row and column of the
        pixel corresponding to the ray.

    Returns:
        Tensor of shape ([num_rays, 3]) containing the bayer mask.
    """

    # Red is top left (0, 0).
    r = (indices[:, 0] % 2 == 0) * (indices[:, 1] % 2 == 0)
    # Green is top right (0, 1) and bottom left (1, 0).
    g = (indices[:, 0] % 2 == 1) * (indices[:, 1] % 2 % 2 == 0) + (
        indices[:, 0] % 2 == 0
    ) * (indices[:, 1] % 2 == 1)
    # Blue is bottom right (1, 1).
    b = (indices[:, 0] % 2 == 1) * (indices[:, 1] % 2 == 1)
    return torch.stack([r, g, b], dim=-1).float()


def save_debug_info(
    weights: torch.Tensor,
    transmittance: torch.Tensor,
    depth: torch.Tensor,
    prop_depth: torch.Tensor,
    step: int,
) -> None:
    """Save output tensors for debugging purposes.

    Args:
        weights: weights tensor.
        transmittance: transmittance tensor.
        depth: depth tensor.
        prop_depth: prop_depth tensor.
        step: current step of training.

    Returns:
        None
    """

    rel_path = "debugging/"

    # if first step create direcrory
    if not os.path.exists(f"{rel_path}debug"):
        os.makedirs(f"{rel_path}debug")

    # save transmittance
    torch.save(transmittance.cpu(), f"{rel_path}debug/transmittance_{step}.pt")

    # save weights
    torch.save(weights.cpu(), f"{rel_path}debug/weights_{step}.pt")

    # save depth
    torch.save(depth.cpu(), f"{rel_path}debug/depth_{step}.pt")

    # save prop depth
    torch.save(prop_depth.cpu(), f"{rel_path}debug/prop_depth_{step}.pt")


def add_water(
    img: Tensor, depth: Tensor, beta_D: Tensor, beta_B: Tensor, B_inf: Tensor
) -> Tensor:
    """Add water effect to image.
       Image formation model from https://openaccess.thecvf.com/content_cvpr_2018/papers/Akkaynak_A_Revised_Underwater_CVPR_2018_paper.pdf (Eq. 20).

    Args:
        img: image to add water effect to.
        beta_D: depth map.
        beta_B: background map.
        B_inf: background image.

    Returns:
        Image with water effect.
    """  # noqa: E501

    depth = depth.repeat_interleave(3, dim=-1)
    I_out = img * torch.exp(-beta_D * depth) + B_inf * (1 - torch.exp(-beta_B * depth))

    return I_out