"""
Template Model — Nerfacto backbone with extensible loss and field hooks.

Supports two backbone strategies:
  1. NerfactoModel (DEFAULT) — Proposal-network sampling + hash encoding.
  2. NeRFModel (VanillaNeRF) — Coarse/fine hierarchical sampling.
To switch, change the parent class and imports.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Type

import torch
from torch import Tensor
from torch.nn import Parameter

from nerfstudio.cameras.rays import RayBundle, RaySamples
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.model_components.losses import MSELoss
from nerfstudio.models.nerfacto import NerfactoModel, NerfactoModelConfig

from .template_field import TemplateField


@dataclass
class TemplateModelConfig(NerfactoModelConfig):
    """Model configuration.

    Inherits all Nerfacto config fields. Override defaults as needed.
    For VanillaNeRF backbone, change parent to VanillaModelConfig.
    """

    _target: Type = field(default_factory=lambda: TemplateModel)

    hash_num_levels: int = 16
    """Number of hash grid resolution levels."""
    hash_features_per_level: int = 2
    """Features per hash grid level."""
    hash_log2_hashmap_size: int = 19
    """Log2 of hash table size (19 = 524K entries)."""
    hash_min_res: int = 16
    """Minimum hash grid resolution."""
    hash_max_res: int = 2048
    """Maximum hash grid resolution."""
    hidden_dim_density: int = 64
    """Hidden dimension of density MLP."""
    hidden_dim_color: int = 64
    """Hidden dimension of color MLP."""
    geo_feat_dim: int = 15
    """Dimension of geometry feature vector passed to color MLP."""

    l2_appearance_mult: float = 0.0
    """L2 regularization weight on appearance embeddings. Proven value: 1e-4."""


class TemplateModel(NerfactoModel):
    """Nerfacto-based model with custom field and losses.

    Override populate_modules() to set up your field, get_loss_dict() to add
    paper-specific losses. get_outputs() usually doesn't need changes with
    the Nerfacto backbone.
    """

    config: TemplateModelConfig

    def populate_modules(self) -> None:
        """Initialize the custom field."""
        super().populate_modules()

        spatial_distortion = None
        if not self.config.disable_scene_contraction:
            spatial_distortion = SceneContraction(order=float("inf"))

        self.field = TemplateField(
            aabb=self.scene_box.aabb,
            num_images=self.num_train_data,
            hash_num_levels=self.config.hash_num_levels,
            hash_features_per_level=self.config.hash_features_per_level,
            hash_log2_hashmap_size=self.config.hash_log2_hashmap_size,
            hash_min_res=self.config.hash_min_res,
            hash_max_res=self.config.hash_max_res,
            num_layers_density=2,
            hidden_dim_density=self.config.hidden_dim_density,
            geo_feat_dim=self.config.geo_feat_dim,
            num_layers_color=3,
            hidden_dim_color=self.config.hidden_dim_color,
            appearance_embedding_dim=self.config.appearance_embed_dim,
            use_appearance_embedding=self.config.use_appearance_embedding,
            use_average_appearance_embedding=self.config.use_average_appearance_embedding,
            spatial_distortion=spatial_distortion,
            average_init_density=self.config.average_init_density,
        )

        self.rgb_loss = MSELoss()

    def get_loss_dict(
        self, outputs: Dict, batch: Dict, metrics_dict=None
    ) -> Dict[str, Tensor]:
        """Compute losses."""
        loss_dict: Dict[str, Tensor] = {}
        image = batch["image"].to(self.device)

        pred_rgb, gt_rgb = self.renderer_rgb.blend_background_for_loss_computation(
            pred_image=outputs["rgb"],
            pred_accumulation=outputs["accumulation"],
            gt_image=image,
        )
        loss_dict["rgb_loss"] = self.rgb_loss(gt_rgb, pred_rgb)

        if self.training and "weights_list" in outputs and "ray_samples_list" in outputs:
            from nerfstudio.model_components.losses import interlevel_loss
            loss_dict["interlevel_loss"] = self.config.interlevel_loss_mult * interlevel_loss(
                outputs["weights_list"], outputs["ray_samples_list"]
            )

        if self.training and metrics_dict is not None and "distortion" in metrics_dict:
            loss_dict["distortion_loss"] = self.config.distortion_loss_mult * metrics_dict["distortion"]

        if self.config.l2_appearance_mult > 0:
            l2 = torch.tensor(0.0, device=self.device)
            emb = getattr(self.field, "embedding_appearance", None)
            if emb is not None:
                num = sum(p.numel() for p in emb.parameters())
                l2 = sum((p**2).sum() for p in emb.parameters()) / max(1, num)
            loss_dict["appearance_l2"] = self.config.l2_appearance_mult * l2

        return loss_dict
