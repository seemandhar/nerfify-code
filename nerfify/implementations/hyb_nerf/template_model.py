"""
Template Model — Nerfacto backbone + Hyb-NeRF hooks
"""
from dataclasses import dataclass, field
from typing import Dict, Type

import torch
from nerfstudio.models.nerfacto import NerfactoModel, NerfactoModelConfig
from nerfstudio.model_components.losses import MSELoss
from nerfstudio.fields.base_field import Field
from nerfstudio.field_components.spatial_distortions import SceneContraction

from nerfify.methods.hyb_nerf.template_field import HybNeRFField

@dataclass
class TemplateModelConfig(NerfactoModelConfig):
    _target: Type = field(default_factory=lambda: TemplateModel)

    coarse_num_frequencies: int = 8
    hash_num_levels: int = 16
    hash_features_per_level: int = 2
    hash_log2_hashmap_size: int = 19
    hash_min_res: int = 180
    hash_max_res: int = 4096
    density_hidden_dim: int = 64
    color_hidden_dim: int = 64
    geo_feat_dim: int = 15
    use_cone_embedding: bool = True
    use_appearance_embedding: bool = False

    l2_appearance_mult: float = 0.0

class TemplateModel(NerfactoModel):

    config: TemplateModelConfig

    def populate_modules(self) -> None:
        super().populate_modules()
        spatial_distortion = None
        if not self.config.disable_scene_contraction:
            spatial_distortion = SceneContraction(order=float("inf"))

        self.field = HybNeRFField(
            aabb=self.scene_box.aabb,
            num_images=self.num_train_data,
            lc=self.config.coarse_num_frequencies,
            lf=self.config.hash_num_levels,
            features_per_level=self.config.hash_features_per_level,
            log2_hashmap_size=self.config.hash_log2_hashmap_size,
            min_res=self.config.hash_min_res,
            max_res=self.config.hash_max_res,
            density_hidden_dim=self.config.density_hidden_dim,
            color_hidden_dim=self.config.color_hidden_dim,
            geo_feat_dim=self.config.geo_feat_dim,
            spatial_distortion=spatial_distortion,
            use_appearance_embedding=self.config.use_appearance_embedding,
            appearance_embedding_dim=self.config.appearance_embed_dim,
            use_average_appearance_embedding=self.config.use_average_appearance_embedding,
            use_cone_embedding=self.config.use_cone_embedding,
        )

        self.rgb_loss = MSELoss()

    def get_loss_dict(self, outputs: Dict, batch: Dict, metrics_dict=None) -> Dict:
        loss_dict: Dict[str, torch.Tensor] = {}
        image = batch["image"].to(self.device)
        loss_dict["rgb_loss"] = self.rgb_loss(image, outputs["rgb"])

        # Optional L2 on appearance embeddings (when present)
        if self.config.l2_appearance_mult > 0:
            l2 = torch.tensor(0.0, device=self.device)
            field: Field = getattr(self, "field", None)  # type: ignore
            emb = getattr(field, "embedding_appearance", None) if field is not None else None
            if emb is not None:
                num = sum(p.numel() for p in emb.parameters())
                l2 = sum((p**2).sum() for p in emb.parameters()) / max(1, num)
            loss_dict["appearance_l2"] = self.config.l2_appearance_mult * l2

        return loss_dict
