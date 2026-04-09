# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Classic NeRF field"""

from typing import Dict, Optional, Tuple, Type

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from nerfstudio.cameras.rays import RaySamples
from nerfstudio.field_components.encodings import Encoding, Identity
from nerfstudio.field_components.field_heads import DensityFieldHead, FieldHead, FieldHeadNames, RGBFieldHead
from nerfstudio.field_components.mlp import MLP
from nerfstudio.field_components.spatial_distortions import SpatialDistortion
from nerfstudio.fields.base_field import Field
from nerfify.methods.anis_nerf.spherical_harmonics import components_from_spherical_harmonics

class AnisMLP(nn.Module):
    def __init__(self, input_enc_ch, hidden, out_ch, num_layers, activation=F.relu):
        super().__init__()
        self.num_layers = num_layers
        self.activation = activation
        self.out_ch = out_ch
        self.linears = nn.ModuleList()
        self.linears.append(nn.Linear(input_enc_ch, hidden))
        for i in range(1, num_layers - 1):
            self.linears.append(nn.Linear(hidden, hidden))
        self.linears.append(nn.Linear(hidden, out_ch))

    def forward(self, enc_input: torch.Tensor) -> torch.Tensor:
        # enc_input: (N, input_enc_ch)
        h = enc_input
        for i, lin in enumerate(self.linears):
            h = lin(h)
            if i < len(self.linears) - 1:
                h = self.activation(h)
        return h

    def get_out_dim(self):
        return self.out_ch


class AnisNeRFField(Field):
    """MiNeRF Field

    Args:
        position_encoding: Position encoder.
        direction_encoding: Direction encoder.
        base_mlp_num_layers: Number of layers for base MLP.
        base_mlp_layer_width: Width of base MLP layers.
        head_mlp_num_layers: Number of layer for output head MLP.
        head_mlp_layer_width: Width of output head MLP layers.
        skip_connections: Where to add skip connection in base MLP.
        use_integrated_encoding: Used integrated samples as encoding input.
        spatial_distortion: Spatial distortion.
    """

    def __init__(
        self,
        position_encoding: Encoding = Identity(in_dim=3),
        direction_encoding: Encoding = Identity(in_dim=3),
        base_mlp_num_layers: int = 8,
        base_mlp_layer_width: int = 128,
        head_mlp_num_layers: int = 2,
        head_mlp_layer_width: int = 128,
        sh_degree: int = 3,
        skip_connections: Tuple[int] = (4,),
        field_heads: Optional[Tuple[Type[FieldHead]]] = (RGBFieldHead,),
        use_integrated_encoding: bool = False,
        spatial_distortion: Optional[SpatialDistortion] = None,
    ) -> None:
        super().__init__()
        self.position_encoding = position_encoding
        self.direction_encoding = direction_encoding
        self.use_integrated_encoding = use_integrated_encoding
        self.spatial_distortion = spatial_distortion
        self.sh_degree = sh_degree

        self.mlp_base = MLP(
            in_dim=self.position_encoding.get_out_dim(),
            num_layers=base_mlp_num_layers,
            layer_width=base_mlp_layer_width,
            skip_connections=skip_connections,
            out_activation=nn.ReLU(),
        )
        self.anis_mlp = AnisMLP(
            input_enc_ch=self.mlp_base.get_out_dim(),
            hidden=base_mlp_layer_width,
            num_layers = 2,
            out_ch = ((1+self.mlp_base.get_out_dim()) * ((1+sh_degree) ** 2)),
            activation=nn.ReLU()
        )
        self.field_output_density = DensityFieldHead(in_dim=self.mlp_base.get_out_dim())

        if field_heads:
            self.mlp_head = MLP(
                in_dim=self.mlp_base.get_out_dim() + self.direction_encoding.get_out_dim(),
                num_layers=head_mlp_num_layers,
                layer_width=head_mlp_layer_width,
                out_activation=nn.ReLU(),
            )
        self.field_heads = nn.ModuleList([field_head() for field_head in field_heads] if field_heads else [])  # type: ignore
        for field_head in self.field_heads:
            field_head.set_in_dim(self.mlp_head.get_out_dim())  # type: ignore

    def get_density(self, ray_samples: RaySamples) -> Tuple[Tensor, Tensor]:
        # for getting density here, we will use the direction here as well. The MLP will be be predicting the coefficients here for the harmonics
        if self.use_integrated_encoding:
            gaussian_samples = ray_samples.frustums.get_gaussian_blob()
            if self.spatial_distortion is not None:
                gaussian_samples = self.spatial_distortion(gaussian_samples)
            encoded_xyz = self.position_encoding(gaussian_samples.mean, covs=gaussian_samples.cov)
        else:
            positions = ray_samples.frustums.get_positions()
            if self.spatial_distortion is not None:
                positions = self.spatial_distortion(positions)
            encoded_xyz = self.position_encoding(positions)
        
        base_mlp_out = self.mlp_base(encoded_xyz)
        coeff = self.anis_mlp(base_mlp_out)
        k_coeff, W_coeff = coeff[:, :, :(self.sh_degree+1)**2], coeff[:, :, (self.sh_degree+1)**2:].reshape(coeff.shape[0], coeff.shape[1], -1, (self.sh_degree+1)**2)
        sh_coeff = components_from_spherical_harmonics(self.sh_degree, ray_samples.frustums.directions)

        density = (k_coeff * sh_coeff).sum(-1)[:, :, None] # (b, N)
        latent = (W_coeff * sh_coeff[:, :, None]).sum(-1) # (b, N, D)
        density = self.field_output_density(base_mlp_out)

        return density, latent, coeff

    def get_outputs(
        self, ray_samples: RaySamples, density_embedding: Optional[Tensor] = None
    ) -> Dict[FieldHeadNames, Tensor]:
        # print(ray_samples.shape)
        outputs = {}
        for field_head in self.field_heads:
            encoded_dir = self.direction_encoding(ray_samples.frustums.directions)
            mlp_out = self.mlp_head(torch.cat([encoded_dir, density_embedding], dim=-1))  # type: ignore
            outputs[field_head.field_head_name] = field_head(mlp_out)
        return outputs

    def forward(self, ray_samples: RaySamples, compute_normals: bool = False) -> Dict[FieldHeadNames, Tensor]:
        """Evaluates the field at points along the ray.

        Args:
            ray_samples: Samples to evaluate field on.
        """
        if compute_normals:
            with torch.enable_grad():
                density, density_embedding, coeff = self.get_density(ray_samples)
        else:
            density, density_embedding, coeff = self.get_density(ray_samples)

        field_outputs = self.get_outputs(ray_samples, density_embedding=density_embedding)
        field_outputs[FieldHeadNames.DENSITY] = density  # type: ignore
        field_outputs["sh_coeff"] = coeff

        if compute_normals:
            with torch.enable_grad():
                normals = self.get_normals()
            field_outputs[FieldHeadNames.NORMALS] = normals  # type: ignore
        return field_outputs