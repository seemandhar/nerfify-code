"""
Template Nerfstudio Field

Extends NerfactoField and (optionally) predicts a direction-only "medium" branch:
- medium_rgb in [0,1]^3 via sigmoid
- medium_bs, medium_attn in R_+^3 via softplus

The medium branch is cheap and can be toggled in the model config.
"""
from typing import Optional
from torch import Tensor
import torch
import torch.nn as nn

from nerfstudio.fields.nerfacto_field import NerfactoField
from nerfstudio.cameras.rays import RaySamples
from nerfstudio.fields.base_field import get_normalized_directions
from nerfstudio.field_components.encodings import SHEncoding
from nerfstudio.field_components.mlp import MLP
from nerfstudio.fields.base_field import Field
from nerfstudio.fields.vanilla_nerf_field import NeRFField


class TemplateNerfField(NeRFField):
    """NerfactoField with optional medium path."""
    aabb: Tensor

    def __init__(self, aabb: Tensor, num_images: int, enable_medium: bool = False) -> None:
        super().__init__(aabb=aabb, num_images=num_images)

    def get_outputs(  # type: ignore[override]
        self, ray_samples: RaySamples, density_embedding: Optional[Tensor] = None
    ):
        # Use Nerfacto outputs first (RGB, DENSITY, etc.)
        outputs = super().get_outputs(ray_samples, density_embedding)

        return outputs
