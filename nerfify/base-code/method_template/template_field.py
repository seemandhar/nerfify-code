"""
Template Nerfstudio Field

Extensible field architecture supporting multiple encoding strategies.
Extend this for your paper's specific field design.
"""
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn

from nerfstudio.cameras.rays import RaySamples
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.field_components.activations import trunc_exp
from nerfstudio.field_components.embedding import Embedding
from nerfstudio.field_components.encodings import HashEncoding, NeRFEncoding, SHEncoding
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.mlp import MLP
from nerfstudio.field_components.spatial_distortions import SpatialDistortion
from nerfstudio.fields.base_field import Field, get_normalized_directions


class TemplateField(Field):
    """NeRF field with hash encoding + SH direction encoding.

    Args:
        aabb: Axis-aligned bounding box of the scene [2, 3].
        num_images: Number of training images (for appearance embedding).
        hash_num_levels: Number of hash grid resolution levels.
        hash_features_per_level: Features stored per hash grid level.
        hash_log2_hashmap_size: Log2 of the hash table size.
        hash_min_res: Minimum hash grid resolution.
        hash_max_res: Maximum hash grid resolution.
        num_layers_density: Number of layers in the density MLP.
        hidden_dim_density: Hidden dimension of the density MLP.
        geo_feat_dim: Dimension of geometry features passed to color MLP.
        num_layers_color: Number of layers in the color MLP.
        hidden_dim_color: Hidden dimension of the color MLP.
        appearance_embedding_dim: Dimension of per-image appearance embedding.
        use_appearance_embedding: Whether to use per-image appearance embedding.
        use_average_appearance_embedding: Use mean embedding at test time (vs zeros).
        spatial_distortion: Spatial distortion to apply (e.g., SceneContraction).
        average_init_density: Scale factor for initial density output.
        implementation: Backend implementation ("tcnn" or "torch").
    """

    aabb: Tensor

    def __init__(
        self,
        aabb: Tensor,
        num_images: int,
        hash_num_levels: int = 16,
        hash_features_per_level: int = 2,
        hash_log2_hashmap_size: int = 19,
        hash_min_res: int = 16,
        hash_max_res: int = 2048,
        num_layers_density: int = 2,
        hidden_dim_density: int = 64,
        geo_feat_dim: int = 15,
        num_layers_color: int = 3,
        hidden_dim_color: int = 64,
        appearance_embedding_dim: int = 32,
        use_appearance_embedding: bool = False,
        use_average_appearance_embedding: bool = True,
        spatial_distortion: Optional[SpatialDistortion] = None,
        average_init_density: float = 1.0,
        implementation: str = "torch",
    ) -> None:
        super().__init__()
        self.register_buffer("aabb", aabb)
        self.geo_feat_dim = geo_feat_dim
        self.spatial_distortion = spatial_distortion
        self.average_init_density = average_init_density

        self.hash_encoding = HashEncoding(
            num_levels=hash_num_levels,
            min_res=hash_min_res,
            max_res=hash_max_res,
            log2_hashmap_size=hash_log2_hashmap_size,
            features_per_level=hash_features_per_level,
            implementation=implementation,
        )
        hash_out_dim = self.hash_encoding.get_out_dim()

        self.direction_encoding = SHEncoding(
            levels=4,
            implementation=implementation,
        )

        self.mlp_density = MLP(
            in_dim=hash_out_dim,
            num_layers=num_layers_density,
            layer_width=hidden_dim_density,
            out_dim=1 + self.geo_feat_dim,
            activation=nn.ReLU(),
            out_activation=None,
            implementation=implementation,
        )

        color_in_dim = (
            self.direction_encoding.get_out_dim()
            + self.geo_feat_dim
            + (appearance_embedding_dim if use_appearance_embedding else 0)
        )
        self.mlp_color = MLP(
            in_dim=color_in_dim,
            num_layers=num_layers_color,
            layer_width=hidden_dim_color,
            out_dim=3,
            activation=nn.ReLU(),
            out_activation=nn.Sigmoid(),
            implementation=implementation,
        )

        self.use_appearance_embedding = use_appearance_embedding
        self.use_average_appearance_embedding = use_average_appearance_embedding
        self.appearance_embedding_dim = appearance_embedding_dim
        if self.use_appearance_embedding:
            self.embedding_appearance = Embedding(num_images, appearance_embedding_dim)

    def _normalize_positions(self, positions: Tensor) -> Tensor:
        """Normalize positions to [0, 1] for hash encoding input."""
        if self.spatial_distortion is not None:
            pos = self.spatial_distortion(positions)
            pos = (pos + 2.0) / 4.0
        else:
            pos = SceneBox.get_normalized_positions(positions, self.aabb)
        selector = ((pos > 0.0) & (pos < 1.0)).all(dim=-1)
        pos = pos * selector[..., None]
        return pos

    def get_density(self, ray_samples: RaySamples) -> Tuple[Tensor, Tensor]:
        """Compute density and geometry features from ray samples.

        Returns:
            density: [*, 1] volume density after activation.
            geo_feat: [*, geo_feat_dim] geometry features for color MLP.
        """
        positions = ray_samples.frustums.get_positions()
        pos_normalized = self._normalize_positions(positions)
        positions_flat = pos_normalized.view(-1, 3)

        hash_features = self.hash_encoding(positions_flat)

        h = self.mlp_density(hash_features).view(*positions.shape[:-1], -1)
        density_before_activation, geo_feat = torch.split(
            h, [1, self.geo_feat_dim], dim=-1
        )

        density = self.average_init_density * trunc_exp(
            density_before_activation.to(positions)
        )

        return density, geo_feat

    def get_outputs(
        self, ray_samples: RaySamples, density_embedding: Optional[Tensor] = None
    ) -> Dict[FieldHeadNames, Tensor]:
        """Compute color from ray samples and geometry features.

        Args:
            ray_samples: Ray sample points.
            density_embedding: Geometry features from get_density() [*, geo_feat_dim].

        Returns:
            Dictionary mapping FieldHeadNames to output tensors.
        """
        assert density_embedding is not None
        outputs: Dict[FieldHeadNames, Tensor] = {}

        directions = get_normalized_directions(ray_samples.frustums.directions)
        dirs_flat = directions.view(-1, 3)
        d_enc = self.direction_encoding(dirs_flat)

        if self.use_appearance_embedding:
            camera_indices = ray_samples.camera_indices.squeeze()
            if self.training:
                embedded_app = self.embedding_appearance(camera_indices)
            else:
                if self.use_average_appearance_embedding:
                    embedded_app = torch.ones(
                        (*directions.shape[:-1], self.appearance_embedding_dim),
                        device=directions.device,
                    ) * self.embedding_appearance.mean(dim=0)
                else:
                    embedded_app = torch.zeros(
                        (*directions.shape[:-1], self.appearance_embedding_dim),
                        device=directions.device,
                    )
            color_in = torch.cat(
                [
                    d_enc,
                    density_embedding.view(-1, self.geo_feat_dim),
                    embedded_app.view(-1, self.appearance_embedding_dim),
                ],
                dim=-1,
            )
        else:
            color_in = torch.cat(
                [d_enc, density_embedding.view(-1, self.geo_feat_dim)], dim=-1
            )

        rgb = self.mlp_color(color_in).view(*directions.shape[:-1], -1).to(directions)
        outputs[FieldHeadNames.RGB] = rgb

        return outputs
