"""
Template Nerfstudio Field

Keeps an optional direction-only "medium" branch example scaffold.
Not used by TVNeRF method but provided as a safe extension point.
"""
from typing import Optional
from torch import Tensor
import torch
import torch.nn as nn

# from nerfstudio.fields.nerfacto_field import NerfactoField
from nerfstudio.fields.vanilla_nerf_field import NeRFField
from nerfstudio.cameras.rays import RaySamples
from nerfstudio.fields.base_field import get_normalized_directions
from nerfstudio.field_components.encodings import SHEncoding
from nerfstudio.field_components.mlp import MLP


class TemplateNerfField(NeRFField):
    """NerfactoField with optional medium path."""
    aabb: Tensor

    def __init__(self, aabb: Tensor, num_images: int, enable_medium: bool = False) -> None:
        super().__init__(aabb=aabb, num_images=num_images)

    def get_outputs(  # type: ignore[override]
        self, ray_samples: RaySamples, density_embedding: Optional[Tensor] = None
    ):
        outputs = super().get_outputs(ray_samples, density_embedding)
