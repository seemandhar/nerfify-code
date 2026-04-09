"""
AR-NeRF Field Implementation
Adaptive Rendering Loss Regularization for Few-Shot NeRF
"""

from typing import Dict, Optional, Tuple, Type

import torch
from torch import Tensor, nn

from nerfstudio.cameras.rays import RaySamples
from nerfstudio.field_components.encodings import Encoding, Identity, NeRFEncoding
from nerfstudio.field_components.field_heads import (
    DensityFieldHead,
    FieldHead,
    FieldHeadNames,
    RGBFieldHead,
)
from nerfstudio.field_components.mlp import MLP
from nerfstudio.fields.vanilla_nerf_field import NeRFField


class ARNeRFField(NeRFField):
    """AR-NeRF Field.

    Extends the vanilla NeRF field to also output per-point color variance (beta^2)
    used for adaptive rendering loss weight learning via uncertainty estimation.

    The MLP outputs (c, beta^2, sigma) instead of just (c, sigma). The variance
    beta^2 is used to compute adaptive loss weights: 1 / beta_bar^2(y).

    Args:
        position_encoding: Position encoder.
        direction_encoding: Direction encoder.
        base_mlp_num_layers: Number of layers for base MLP.
        base_mlp_layer_width: Width of base MLP layers.
        head_mlp_num_layers: Number of layers for output head MLP.
        head_mlp_layer_width: Width of output head MLP layers.
        skip_connections: Where to add skip connection in base MLP.
    """

    def __init__(
        self,
        position_encoding: Encoding = Identity(in_dim=3),
        direction_encoding: Encoding = Identity(in_dim=3),
        base_mlp_num_layers: int = 8,
        base_mlp_layer_width: int = 256,
        head_mlp_num_layers: int = 2,
        head_mlp_layer_width: int = 128,
        skip_connections: Tuple[int, ...] = (4,),
        field_heads: Optional[Tuple[Type[FieldHead], ...]] = (RGBFieldHead,),
    ) -> None:
        super().__init__(
            position_encoding=position_encoding,
            direction_encoding=direction_encoding,
            base_mlp_num_layers=base_mlp_num_layers,
            base_mlp_layer_width=base_mlp_layer_width,
            head_mlp_num_layers=head_mlp_num_layers,
            head_mlp_layer_width=head_mlp_layer_width,
            skip_connections=skip_connections,
            field_heads=field_heads,
        )

        # Extra head to predict log(beta^2) per point (uncertainty/variance).
        # We predict log-variance for numerical stability, then exponentiate.
        # Output dim = 1 (scalar variance per point).
        self.beta_head = nn.Linear(base_mlp_layer_width, 1)
        nn.init.xavier_uniform_(self.beta_head.weight)
        nn.init.zeros_(self.beta_head.bias)

    def get_density(self, ray_samples: RaySamples) -> Tuple[Tensor, Tensor]:
        """Override to also store base MLP output for beta prediction."""
        positions = ray_samples.frustums.get_positions()
        encoded_xyz = self.position_encoding(positions)
        base_mlp_out = self.mlp_base(encoded_xyz)
        density = self.field_output_density(base_mlp_out)
        return density, base_mlp_out

    def forward(self, ray_samples: RaySamples) -> Dict[FieldHeadNames, Tensor]:
        """Evaluates the field at points along the ray.

        Returns density, RGB, and per-point variance beta^2.
        """
        density, base_mlp_out = self.get_density(ray_samples)

        # Predict log(beta^2) and exponentiate to get beta^2 >= 0.
        # Clamp for stability: beta^2 in [exp(-10), exp(10)].
        log_beta2 = self.beta_head(base_mlp_out)
        beta2 = torch.exp(log_beta2.clamp(-10.0, 10.0))  # (N_pts, 1)

        field_outputs = self.get_outputs(ray_samples, density_embedding=base_mlp_out)
        field_outputs[FieldHeadNames.DENSITY] = density

        # Store beta^2 under a custom key accessed by the model.
        # We reuse UNCERTAINTY if available, else store in a plain key.
        field_outputs["beta2"] = beta2  # type: ignore

        return field_outputs