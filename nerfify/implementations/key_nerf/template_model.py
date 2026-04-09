"""
Template Model — Nerfacto backbone + Informative Rays hooks
"""
from dataclasses import dataclass, field
from typing import Dict, Type

import torch
# from nerfstudio.models.nerfacto import NerfactoModel, NerfactoModelConfig
from nerfstudio.models.vanilla_nerf import NeRFModel, VanillaModelConfig

@dataclass
class TemplateModelConfig(VanillaModelConfig):
    """Config for TemplateModel (inherits all Nerfacto knobs)."""
    _target: Type = field(default_factory=lambda: TemplateModel)

class TemplateModel(NeRFModel):
    """Nerfacto-based model with minimal, safe extensions."""

    config: TemplateModelConfig

    def populate_modules(self) -> None:
        """Keep Nerfacto modules (proposal nets, field, renderers, losses, metrics)."""
        super().populate_modules()

    def get_loss_dict(self, outputs: Dict, batch: Dict, metrics_dict=None) -> Dict:
        """Start with Nerfacto losses; add optional regularizers."""
        loss_dict = super().get_loss_dict(outputs, batch, metrics_dict)

        return loss_dict
