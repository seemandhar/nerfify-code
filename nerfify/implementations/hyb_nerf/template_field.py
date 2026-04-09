from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn

from nerfstudio.cameras.rays import RaySamples
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.field_components.activations import trunc_exp
from nerfstudio.field_components.embedding import Embedding
from nerfstudio.field_components.encodings import HashEncoding, SHEncoding, NeRFEncoding
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.mlp import MLP
from nerfstudio.field_components.spatial_distortions import SpatialDistortion
from nerfstudio.fields.base_field import Field, get_normalized_directions


def _upper_triangular_from_cov(sigma: Tensor) -> Tensor:
    s_xx = sigma[..., 0, 0]
    s_xy = sigma[..., 0, 1]
    s_xz = sigma[..., 0, 2]
    s_yy = sigma[..., 1, 1]
    s_yz = sigma[..., 1, 2]
    s_zz = sigma[..., 2, 2]
    return torch.stack([s_xx, s_xy, s_xz, s_yy, s_yz, s_zz], dim=-1)


class HybNeRFField(Field):
    aabb: Tensor

    def __init__(
        self,
        aabb: Tensor,
        num_images: int,
        lc: int = 8,
        lf: int = 16,
        features_per_level: int = 2,
        log2_hashmap_size: int = 19,
        min_res: int = 180,
        max_res: int = 4096,
        density_hidden_dim: int = 64,
        color_hidden_dim: int = 64,
        geo_feat_dim: int = 15,
        spatial_distortion: Optional[SpatialDistortion] = None,
        use_appearance_embedding: bool = True,
        appearance_embedding_dim: int = 32,
        use_average_appearance_embedding: bool = True,
        use_cone_embedding: bool = True,
    ) -> None:
        super().__init__()
        self.register_buffer("aabb", aabb)
        self.geo_feat_dim = geo_feat_dim
        self.spatial_distortion = spatial_distortion

        self.hash_enc = HashEncoding(
            num_levels=lf,
            min_res=min_res,
            max_res=max_res,
            log2_hashmap_size=log2_hashmap_size,
            features_per_level=features_per_level,
            implementation="torch",
        )
        self.fine_out_dim = self.hash_enc.get_out_dim()

        self.pos_fourier = NeRFEncoding(
            in_dim=3, num_frequencies=lc, min_freq_exp=0.0, max_freq_exp=float(lc - 1), include_input=False
        )
        self.cone_fourier = NeRFEncoding(
            in_dim=6, num_frequencies=lc, min_freq_exp=0.0, max_freq_exp=float(lc - 1), include_input=False
        )
        self.coarse_out_dim = self.pos_fourier.get_out_dim()

        self.alpha_mlp = MLP(
            in_dim=self.fine_out_dim + self.cone_fourier.get_out_dim(),
            num_layers=2,
            layer_width=64,
            out_dim=self.coarse_out_dim,
            activation=nn.ReLU(),
            out_activation=nn.Tanh(),
            implementation="torch",
        )

        self.direction_encoding = SHEncoding(levels=4, implementation="torch")

        self.density_mlp = MLP(
            in_dim=self.coarse_out_dim + self.fine_out_dim,
            num_layers=2,  # single hidden layer
            layer_width=density_hidden_dim,
            out_dim=1 + self.geo_feat_dim,
            activation=nn.ReLU(),
            out_activation=None,
            implementation="torch",
        )
        self.color_mlp = MLP(
            in_dim=self.direction_encoding.get_out_dim() + self.geo_feat_dim + (appearance_embedding_dim if use_appearance_embedding else 0),
            num_layers=3,  # two hidden layers
            layer_width=color_hidden_dim,
            out_dim=3,
            activation=nn.ReLU(),
            out_activation=nn.Sigmoid(),
            implementation="torch",
        )

        self.use_appearance_embedding = use_appearance_embedding
        self.use_average_appearance_embedding = use_average_appearance_embedding
        self.appearance_embedding_dim = appearance_embedding_dim
        if self.use_appearance_embedding:
            self.embedding_appearance = Embedding(num_images, appearance_embedding_dim)

        self.use_cone_embedding = use_cone_embedding

    def _normalize_positions(self, positions: Tensor) -> Tensor:
        if self.spatial_distortion is not None:
            pos = self.spatial_distortion(positions)
            # map from [-2,2] to [0,1]
            pos = (pos + 2.0) / 4.0
        else:
            pos = SceneBox.get_normalized_positions(positions, self.aabb)
        # ensure within [0,1]
        selector = ((pos > 0.0) & (pos < 1.0)).all(dim=-1)
        pos = pos * selector[..., None]
        return pos

    def _cone_cov_upper(self, ray_samples: RaySamples) -> Tensor:
        d = get_normalized_directions(ray_samples.frustums.directions)
        ddT = d[..., :, None] * d[..., None, :]  # [..., 3, 3]
        eye = torch.eye(3, device=d.device, dtype=d.dtype).view(*(1,) * (d.dim() - 1), 3, 3)

        if hasattr(ray_samples, "deltas") and ray_samples.deltas is not None:
            delta = ray_samples.deltas
        else:
            delta = (ray_samples.frustums.ends - ray_samples.frustums.starts)
        sigma_t2 = (delta[..., 0:1] ** 2) / 12.0  # variance of uniform segment

        sigma_r2 = ray_samples.frustums.pixel_area[..., 0:1]

        Sigma = sigma_t2[..., None] * ddT + sigma_r2[..., None] * (eye - ddT)
        tri = _upper_triangular_from_cov(Sigma)  # [..., 6]
        return tri

    def get_density(self, ray_samples: RaySamples) -> Tuple[Tensor, Tensor]:
        positions = ray_samples.frustums.get_positions()
        pos_hash = self._normalize_positions(positions)
        pos_flat = positions.view(-1, 3)
        pos_fourier = self.pos_fourier(pos_flat)  # [N, D_pos]

        fine_flat = self.hash_enc(pos_hash.view(-1, 3))  # [N, D_fine]

        if self.use_cone_embedding:
            tri = self._cone_cov_upper(ray_samples)  # [..., 6]
            cone_fourier = self.cone_fourier(tri.view(-1, 6))  # [N, D_cone]
        else:
            cone_fourier = torch.zeros_like(self.cone_fourier(torch.zeros_like(pos_flat[:, :6])))

        alpha_in = torch.cat([fine_flat, cone_fourier], dim=-1)
        alpha = self.alpha_mlp(alpha_in)  # [N, D_pos]
        coarse_feat = pos_fourier * alpha

        hybrid = torch.cat([coarse_feat, fine_flat], dim=-1)
        sigma_geo = self.density_mlp(hybrid).view(*positions.shape[:-1], -1)
        density_before_activation, geo = torch.split(sigma_geo, [1, self.geo_feat_dim], dim=-1)

        density = trunc_exp(density_before_activation.to(positions))
        return density, geo

    def get_outputs(self, ray_samples: RaySamples, density_embedding: Optional[Tensor] = None) -> Dict[FieldHeadNames, Tensor]:
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
                        (*directions.shape[:-1], self.appearance_embedding_dim), device=directions.device
                    ) * self.embedding_appearance.mean(dim=0)
                else:
                    embedded_app = torch.zeros(
                        (*directions.shape[:-1], self.appearance_embedding_dim), device=directions.device
                    )
            color_in = torch.cat([d_enc, density_embedding.view(-1, self.geo_feat_dim), embedded_app.view(-1, self.appearance_embedding_dim)], dim=-1)
        else:
            color_in = torch.cat([d_enc, density_embedding.view(-1, self.geo_feat_dim)], dim=-1)

        rgb = self.color_mlp(color_in).view(*directions.shape[:-1], -1).to(directions)
        outputs[FieldHeadNames.RGB] = rgb
        return outputs
