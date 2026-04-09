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
from nerfify.methods.ers.enerf.enerf_field import ENeRFField

from nerfstudio.models.base_model import Model, ModelConfig  # for custom Model
from .network import Network
from .losses.enerf import LossAndMetrics

@dataclass
class ENeRFModelConfig(ModelConfig):
    """ENeRF Model Configuration."""

    _target: Type = field(default_factory=lambda: ENeRFModel)

    # num_coarse_samples: int = 32 #16
    # num_importance_samples: int = 64 #32
    # eval_num_rays_per_chunk: int = 4096

class ENeRFModel(Model):
    """ENeRF Model."""

    config: ENeRFModelConfig

    def __init__(
        self,
        config: ENeRFModelConfig,
        **kwargs
    ) -> None:
        self.field_coarse = None
        self.field_fine = None
        self.temporal_distortion = None

        super().__init__(
            config=config,
            scene_box=None,
            **kwargs,
        )

        self.loss_and_metrics = LossAndMetrics()

    def compute_loss_and_metrics(self, model_outputs, batch):
        return self.loss_and_metrics(model_outputs, batch)

    def populate_modules(self):
        super().populate_modules()

        # fields
        self.network = Network()

    def get_param_groups(self):
        return {"fields" : list(self.network.parameters())}
    
    def forward(self, batch):
        return self.network(batch)