"""
Template Nerfstudio Field

Currently this subclasses the NerfactoField. Consider subclassing the base Field.
"""

from typing import Dict, Literal, Tuple, Optional, Type

import torch
from torch import Tensor, nn

from nerfstudio.cameras.rays import Frustums, RaySamples
from nerfstudio.field_components.spatial_distortions import SpatialDistortion
from nerfstudio.fields.vanilla_nerf_field import NeRFField  # for subclassing NerfactoField
from nerfstudio.fields.base_field import Field  # for custom Field

from nerfstudio.field_components.encodings import Encoding, Identity
from nerfstudio.field_components.field_heads import DensityFieldHead, FieldHead, FieldHeadNames, RGBFieldHead
from nerfstudio.field_components.mlp import MLP


class ENeRFField(NeRFField):
    """ENeRF Field

    Args:
    """

    def __init__(
        self,
    ) -> None:
        super().__init__()


    def forward(self, ray_samples: RaySamples) -> Dict[FieldHeadNames, Tensor]:
        """Evaluates the field at points along the ray.

        Args:
            ray_samples: Samples to evaluate field on.
        """
        pass