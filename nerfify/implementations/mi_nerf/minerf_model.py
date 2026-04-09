"""
Template Model File

Currently this subclasses the Nerfacto model. Consider subclassing from the base Model.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Type

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.configs.config_utils import to_immutable_dict
from nerfstudio.field_components.temporal_distortions import TemporalDistortionKind
from nerfstudio.field_components.encodings import NeRFEncoding
from nerfify.methods.mi_nerf.minerf_field import MiNeRFField

from nerfstudio.models.vanilla_nerf import NeRFModel, VanillaModelConfig  # for subclassing Nerfacto model
from nerfstudio.models.base_model import Model, ModelConfig  # for custom Model


@dataclass
class MiNeRFModelConfig(VanillaModelConfig):
    """BioNeRF Model Configuration."""

    _target: Type = field(default_factory=lambda: MiNeRFModel)

    num_coarse_samples: int = 32 #16
    num_importance_samples: int = 64 #32
    # eval_num_rays_per_chunk: int = 4096

class MiNeRFModel(NeRFModel):
    """MiNeRF Model."""

    config: MiNeRFModelConfig

    def __init__(
        self,
        config: MiNeRFModelConfig,
        **kwargs,
    ) -> None:
        self.field_coarse = None
        self.field_fine = None
        self.temporal_distortion = None

        super().__init__(
            config=config,
            **kwargs,
        )

    def populate_modules(self):
        super().populate_modules()

        # fields
        position_encoding = NeRFEncoding(
            in_dim=3, num_frequencies=10, min_freq_exp=0.0, max_freq_exp=8.0, include_input=True
        )
        direction_encoding = NeRFEncoding(
            in_dim=3, num_frequencies=4, min_freq_exp=0.0, max_freq_exp=4.0, include_input=True
        )

        self.field_coarse = MiNeRFField(position_encoding=position_encoding, direction_encoding=direction_encoding)
        self.field_fine = MiNeRFField(position_encoding=position_encoding, direction_encoding=direction_encoding)

    # def get_outputs(self, ray_bundle : RayBundle):
    #     print(ray_bundle.shape)
    #     exit(0)