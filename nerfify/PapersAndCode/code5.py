"bionerf.py"

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
from bionerf.bionerf_field import BioNeRFField

from nerfstudio.models.vanilla_nerf import NeRFModel, VanillaModelConfig  # for subclassing Nerfacto model
from nerfstudio.models.base_model import Model, ModelConfig  # for custom Model


@dataclass
class BioNeRFModelConfig(VanillaModelConfig):
    """BioNeRF Model Configuration."""

    _target: Type = field(default_factory=lambda: BioNeRFModel)

    num_coarse_samples: int = 32 #16
    num_importance_samples: int = 64 #32
    # eval_num_rays_per_chunk: int = 4096

class BioNeRFModel(NeRFModel):
    """BioNeRF Model."""

    config: BioNeRFModelConfig

    def __init__(
        self,
        config: BioNeRFModelConfig,
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

        self.field_coarse = BioNeRFField(
            position_encoding=position_encoding,
            direction_encoding=direction_encoding,
        )

        self.field_fine = BioNeRFField(
            position_encoding=position_encoding,
            direction_encoding=direction_encoding,
        )

"bionerf_config.py"
from __future__ import annotations

from bionerf.bionerf_datamanager import BioNeRFDataManagerConfig
from bionerf.bionerf import BioNeRFModelConfig
from bionerf.bionerf_pipeline import BioNeRFPipelineConfig
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig, RAdamOptimizerConfig
from nerfstudio.engine.schedulers import (
    ExponentialDecaySchedulerConfig,
)
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.plugins.types import MethodSpecification




bionerf_method = MethodSpecification(
    config=TrainerConfig(
        method_name="bionerf", 
        steps_per_eval_batch=500,
        steps_per_save=2000,
        max_num_iterations=26000,
        mixed_precision=True,
        pipeline=BioNeRFPipelineConfig(
            datamanager=BioNeRFDataManagerConfig(
                dataparser=NerfstudioDataParserConfig(),
                train_num_rays_per_batch=4096, #4096
                eval_num_rays_per_batch=4096, #4096
            ),
            model=BioNeRFModelConfig(
                eval_num_rays_per_chunk=4096,
            ),
            # model=BioNeRFModelConfig(
            #     eval_num_rays_per_chunk=1 << 15,
            # ),
        ),
        optimizers={
            # TODO: consider changing optimizers depending on your custom method
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=5e-4, eps=1e-15), #lr=5e-4,, eps=1e-15
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
            },
            "fields": {
                "optimizer": RAdamOptimizerConfig(lr=5e-4, eps=1e-15),#lr=5e-4, eps=1e-15
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=50000),
            },
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15), #lr=1e-3, eps=1e-15
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=5000),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="wandb", #"viewer+wandb"
    ),
    description="Implementation for BioNeRF.",
)


"bionerf_datamanager.py"
"""
Template DataManager
"""

from dataclasses import dataclass, field
from typing import Dict, Literal, Tuple, Type, Union

import torch

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.datamanagers.base_datamanager import (
    VanillaDataManager,
    VanillaDataManagerConfig,
)


@dataclass
class BioNeRFDataManagerConfig(VanillaDataManagerConfig):
    """Template DataManager Config

    Add your custom datamanager config parameters here.
    """

    _target: Type = field(default_factory=lambda: BioNeRFDataManager)


class BioNeRFDataManager(VanillaDataManager):
    """BioNeRF DataManager

    Args:
        config: the DataManagerConfig used to instantiate class
    """

    config: BioNeRFDataManagerConfig

    def __init__(
        self,
        config: BioNeRFDataManagerConfig,
        device: Union[torch.device, str] = "cpu",
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        **kwargs,  # pylint: disable=unused-argument
    ):
        super().__init__(
            config=config, device=device, test_mode=test_mode, world_size=world_size, local_rank=local_rank, **kwargs
        )

    def next_train(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the train dataloader."""
        self.train_count += 1
        image_batch = next(self.iter_train_image_dataloader)
        assert self.train_pixel_sampler is not None
        assert isinstance(image_batch, dict)
        batch = self.train_pixel_sampler.sample(image_batch)
        ray_indices = batch["indices"]
        ray_bundle = self.train_ray_generator(ray_indices)
        return ray_bundle, batch
    
"bionerf_field.py"
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


class BioNeRFField(NeRFField):
    """BioNeRF Field

    Args:
        position_encoding: Position encoder.
        direction_encoding: Direction encoder.
        base_mlp_delta_num_layers: Number of layers for Delta MLP.
        base_mlp_c_num_layers: Number of layers for c MLP.
        head_mlp_delta_num_layers: Number of layers for Delta' MLP.
        head_mlp_c_num_layers: Number of layers for c' MLP.
        base_mlp_width: Width of Delta and c MLP layers.
        head_mlp_delta_width: Width of Delta' layers.
        head_mlp_c_width: Width of c' layers.
    """

    def __init__(
        self,
        position_encoding: Encoding = Identity(in_dim=3),
        direction_encoding: Encoding = Identity(in_dim=3),
        base_mlp_delta_num_layers: int = 3,
        base_mlp_c_num_layers: int = 3,
        base_mlp_width: int = 256,
        head_mlp_delta_num_layers: int = 2,
        head_mlp_delta_width: int = 256,
        head_mlp_c_num_layers: int = 1,
        head_mlp_c_width: int = 128,
        field_heads: Optional[Tuple[Type[FieldHead]]] = (RGBFieldHead,)
    ) -> None:
        super().__init__()
        self.position_encoding = position_encoding
        self.direction_encoding = direction_encoding

        self.mlp_base_delta = MLP(
            in_dim=self.position_encoding.get_out_dim(),
            num_layers=base_mlp_delta_num_layers,
            layer_width=base_mlp_width,
            out_activation=nn.ReLU(),
        )

        self.mlp_head_delta = MLP(
            in_dim=self.position_encoding.get_out_dim() + base_mlp_width,
            num_layers=head_mlp_delta_num_layers,
            layer_width=head_mlp_delta_width,
            out_activation=nn.ReLU(),
        )

        self.mlp_base_c = MLP(
            in_dim=self.position_encoding.get_out_dim(),
            num_layers=base_mlp_c_num_layers,
            layer_width=base_mlp_width,
            out_activation=nn.ReLU(),
        )

        self.W_gamma = nn.Linear(base_mlp_width*2, base_mlp_width)
        self.W_psi = nn.Linear(base_mlp_width*2, base_mlp_width)
        self.W_mu = nn.Linear(base_mlp_width*2, base_mlp_width)
        self.memory = None

        self.field_output_density = DensityFieldHead(in_dim=self.mlp_head_delta.get_out_dim())

        if field_heads:
            self.mlp_head_c = MLP(
                in_dim=self.direction_encoding.get_out_dim() + base_mlp_width,
                num_layers=head_mlp_c_num_layers,
                layer_width=head_mlp_c_width,
                out_activation=nn.ReLU(),
            )

        self.field_heads = nn.ModuleList([field_head() for field_head in field_heads] if field_heads else [])  # type: ignore
        for field_head in self.field_heads:
            field_head.set_in_dim(self.mlp_head_c.get_out_dim())  # type: ignore


    def forward(self, ray_samples: RaySamples) -> Dict[FieldHeadNames, Tensor]:
        """Evaluates the field at points along the ray.

        Args:
            ray_samples: Samples to evaluate field on.
        """
        positions = ray_samples.frustums.get_positions()
        encoded_xyz = self.position_encoding(positions)

        h_delta = self.mlp_base_delta(encoded_xyz)  
        h_c = self.mlp_base_c(encoded_xyz)     

        # memory
        f_delta = torch.sigmoid(h_delta)
        f_c = torch.sigmoid(h_c)
        h_delta_h_c = torch.cat((h_delta, h_c), dim=-1)

        gamma = torch.tanh(self.W_gamma(h_delta_h_c))
        f_mu = torch.sigmoid(self.W_mu(h_delta_h_c))
        mu = torch.mul(gamma, f_mu)

        if self.memory is None:
            memory = torch.tanh(mu)
            self.memory = memory.detach()
        else:
            if self.memory.shape[0]!=ray_samples.frustums.get_positions().shape[0]:
                f_psi = torch.sigmoid(self.W_psi(h_delta_h_c))

                memory = torch.tanh(mu + torch.mul(f_psi, self.memory[:encoded_xyz.shape[0],:]))
                new_memory = self.memory.clone()
                new_memory[:memory.shape[0],:] = memory
                self.memory = new_memory.detach()

                del new_memory      
                torch.cuda.empty_cache()    


            else:
                f_psi = torch.sigmoid(self.W_psi(h_delta_h_c))
                memory = torch.tanh(mu + torch.mul(f_psi, self.memory))
                self.memory = memory.detach()    

        density = self.get_density(encoded_xyz, torch.mul(f_delta, memory))


        # ------------ ate aqui ------------------

        field_outputs = self.get_outputs(ray_samples, torch.mul(f_c, memory))
        field_outputs[FieldHeadNames.DENSITY] = density  # type: ignore

        return field_outputs

    def get_density(self, encoded_xyz: Tensor, density_embedding: Tensor) -> Tensor:
        base_mlp_out = self.mlp_head_delta(torch.cat([encoded_xyz, density_embedding], dim=-1))
        return self.field_output_density(base_mlp_out)
    
    def get_outputs(
        self, ray_samples: RaySamples, density_embedding: Tensor) -> Dict[FieldHeadNames, Tensor]:
        outputs = {}
        for field_head in self.field_heads:
            encoded_dir = self.direction_encoding(ray_samples.frustums.directions)
            mlp_out = self.mlp_head_c(torch.cat([encoded_dir, density_embedding], dim=-1))  # type: ignore
            outputs[field_head.field_head_name] = field_head(mlp_out)
        return outputs
    
"bionerf_pipeline.py"

"""
Nerfstudio Template Pipeline
"""

import typing
from dataclasses import dataclass, field
from typing import Literal, Optional, Type

import torch.distributed as dist
from torch.cuda.amp.grad_scaler import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP

from bionerf.bionerf_datamanager import BioNeRFDataManagerConfig
from bionerf.bionerf import BioNeRFModelConfig, BioNeRFModel
from nerfstudio.data.datamanagers.base_datamanager import (
    DataManager,
    DataManagerConfig,
)
from nerfstudio.models.base_model import ModelConfig
from nerfstudio.pipelines.base_pipeline import (
    VanillaPipeline,
    VanillaPipelineConfig,
)


@dataclass
class BioNeRFPipelineConfig(VanillaPipelineConfig):
    """Configuration for pipeline instantiation"""

    _target: Type = field(default_factory=lambda: BioNeRFPipeline)
    """target class to instantiate"""
    datamanager: DataManagerConfig = field(default_factory=BioNeRFDataManagerConfig)
    """specifies the datamanager config"""
    model: ModelConfig = field(default_factory=BioNeRFModelConfig)
    """specifies the model config"""


class BioNeRFPipeline(VanillaPipeline):
    """BioNeRF Pipeline

    Args:
        config: the pipeline config used to instantiate class
    """

    def __init__(
        self,
        config: BioNeRFPipelineConfig,
        device: str,
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        grad_scaler: Optional[GradScaler] = None,
    ):
        super(VanillaPipeline, self).__init__()
        self.config = config
        self.test_mode = test_mode
        self.datamanager: DataManager = config.datamanager.setup(
            device=device, test_mode=test_mode, world_size=world_size, local_rank=local_rank
        )
        self.datamanager.to(device)

        assert self.datamanager.train_dataset is not None, "Missing input dataset"
        self._model = config.model.setup(
            scene_box=self.datamanager.train_dataset.scene_box,
            num_train_data=len(self.datamanager.train_dataset),
            metadata=self.datamanager.train_dataset.metadata,
            device=device,
            grad_scaler=grad_scaler,
        )
        self.model.to(device)

        self.world_size = world_size
        if world_size > 1:
            self._model = typing.cast(
                BioNeRFModel, DDP(self._model, device_ids=[local_rank], find_unused_parameters=True)
            )
            dist.barrier(device_ids=[local_rank])