"zipnerf_config.py"

"""
Nerfstudio ZipNerf Config

Define your custom method here that registers with Nerfstudio CLI.
"""

from __future__ import annotations

from zipnerf_ns.zipnerf_datamanager import (
    ZipNerfDataManagerConfig,
)
from zipnerf_ns.zipnerf_model import ZipNerfModelConfig
from zipnerf_ns.zipnerf_pipeline import (
    ZipNerfPipelineConfig,
)
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.dataparsers.colmap_dataparser import ColmapDataParserConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import (
    ExponentialDecaySchedulerConfig,
)
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.plugins.types import MethodSpecification


zipnerf_method = MethodSpecification(
    config=TrainerConfig(
        method_name="zipnerf", 
        steps_per_eval_batch=1000,
        steps_per_eval_image=5000,
        steps_per_save=5000,
        max_num_iterations=25000,
        mixed_precision=True,
        log_gradients=False,
        pipeline=ZipNerfPipelineConfig(
            datamanager=ZipNerfDataManagerConfig(
                dataparser=ColmapDataParserConfig(downscale_factor=4,orientation_method="up",center_method="poses", colmap_path="sparse/0"),
                train_num_rays_per_batch=8192,
                eval_num_rays_per_batch=8192,
            ),
            model=ZipNerfModelConfig(
                eval_num_rays_per_chunk=1 << 15,
                gin_file=["configs/360.gin"],
                proposal_weights_anneal_max_num_iters=1000,
            ),
        ),
        optimizers={
            "model": {
                "optimizer": AdamOptimizerConfig(lr=8e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(warmup_steps=1000,lr_final=1e-3, max_steps=25000)
            }
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="An unofficial pytorch implementation of 'Zip-NeRF: Anti-Aliased Grid-Based Neural Radiance Fields' https://arxiv.org/abs/2304.06706. ",
)



"""
ZipNerf DataManager
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
class ZipNerfDataManagerConfig(VanillaDataManagerConfig):
    """ZipNerf DataManager Config

    Add your custom datamanager config parameters here.
    """

    _target: Type = field(default_factory=lambda: ZipNerfDataManager)


class ZipNerfDataManager(VanillaDataManager):
    """ZipNerf DataManager

    Args:
        config: the DataManagerConfig used to instantiate class
    """

    config: ZipNerfDataManagerConfig

    def __init__(
        self,
        config: ZipNerfDataManagerConfig,
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
        batch['rgb'] = batch['image'].to(self.device)
        ray_indices = batch["indices"]
        ray_bundle = self.train_ray_generator(ray_indices)
        return ray_bundle, batch
    

    def next_eval(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the eval dataloader."""
        self.eval_count += 1
        image_batch = next(self.iter_eval_image_dataloader)
        assert self.eval_pixel_sampler is not None
        assert isinstance(image_batch, dict)
        batch = self.eval_pixel_sampler.sample(image_batch)
        batch['rgb'] = batch['image'].to(self.device)
        ray_indices = batch["indices"]
        ray_bundle = self.eval_ray_generator(ray_indices)
        return ray_bundle, batch
    

"zipnerf_model.py"

from dataclasses import dataclass, field
import importlib
import os
from typing import Dict, List, Literal, Type
from nerfstudio.models.base_model import Model, ModelConfig  # for custom Model
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.model_components.scene_colliders import NearFarCollider
from nerfstudio.model_components.renderers import RGBRenderer
from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Type
import torch
from torch.nn import Parameter
from internal import train_utils
from internal.configs import Config
from internal.models import Model as zipnerf
import gin
import numpy as np
from nerfstudio.utils import colormaps
@dataclass
class ZipNerfModelConfig(ModelConfig):
    gin_file: list = None 
    """Config files list to load default setting of Model/NerfMLP/PropMLP as zipnerf-pytorch"""
    compute_extras: bool = True
    """if True, compute extra quantities besides color."""
    proposal_weights_anneal_max_num_iters: int = 1000  
    """Max num iterations for the annealing function. Set to the value of max_train_iterations to have same behavior as zipnerf-pytorch."""
    rand: bool = True
    """random number generator (or None for deterministic output)."""
    zero_glo: bool = False
    """if True, when using GLO pass in vector of zeros."""
    background_color: Literal["random", "black", "white"] = "white"
    """Whether to randomize the background color."""
    _target: Type = field(default_factory=lambda: ZipNerfModel)

class ZipNerfModel(Model):
    config: ZipNerfModelConfig

    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()
        # update default setting
        # gin.parse_config_files_and_bindings(self.config.gin_file, None)
        gin_files = []
        for g in self.config.gin_file:
            if os.path.exists(g):
                gin_files.append(g)
            else:
                package_path = importlib.util.find_spec("zipnerf_ns").origin.split('/')[:-2]
                package_path = '/'.join(package_path)
                gin_files.append(package_path+'/'+g)
        gin.parse_config_files_and_bindings(gin_files, None)
        config = Config()

        self.zipnerf = zipnerf(config=config)
        
        self.collider = NearFarCollider(near_plane=self.zipnerf.config.near, far_plane=self.zipnerf.config.far)
        self.step = 0

        # Renderer
        self.renderer_rgb = RGBRenderer(background_color=self.config.background_color)

        # Metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)

    def construct_batch_from_raybundle(self, ray_bundle):
        batch = {}
        batch['origins'] = ray_bundle.origins
        batch['directions'] = ray_bundle.directions * ray_bundle.metadata["directions_norm"]
        batch['viewdirs'] = ray_bundle.directions
        batch['radii'] = torch.sqrt(ray_bundle.pixel_area)* 2 / torch.sqrt(torch.full_like(ray_bundle.pixel_area,12.))
        batch['cam_idx'] = ray_bundle.camera_indices
        batch['near'] = ray_bundle.nears
        batch['far'] = ray_bundle.fars
        batch['cam_dirs'] = None  # did not be calculated in raybundle
        # batch['imageplane'] = None
        # batch['exposure_values'] = None
        return batch
    
    def get_outputs(self, ray_bundle: RayBundle):
        ray_bundle.metadata["viewdirs"] = ray_bundle.directions
        ray_bundle.metadata["radii"] = torch.sqrt(ray_bundle.pixel_area)* 2 / torch.sqrt(torch.full_like(ray_bundle.pixel_area,12.))
        ray_bundle.directions = ray_bundle.directions * ray_bundle.metadata["directions_norm"]
        
        if self.training:
            anneal_frac = np.clip(self.step / self.config.proposal_weights_anneal_max_num_iters, 0, 1)
        else:
            anneal_frac = 1.0
        batch = self.construct_batch_from_raybundle(ray_bundle)

        renderings, ray_history = self.zipnerf(
                rand=self.config.rand if self.training else False,  # set to false when evaluating or rendering
                batch=batch,
                train_frac=anneal_frac, 
                compute_extras=self.config.compute_extras,
                zero_glo=self.config.zero_glo if self.training else True) # set to True when evaluating or rendering
        
        outputs={}

        # showed by viewer
        outputs['rgb']=renderings[2]['rgb']
        outputs['depth']=renderings[2]['depth'].unsqueeze(-1)
        outputs['accumulation']=renderings[2]['acc']
        if self.config.compute_extras:
            outputs['distance_mean']=renderings[2]['distance_mean']
            outputs['distance_median']=renderings[2]['distance_median']

        # for loss calculation
        outputs['renderings']=renderings
        outputs['ray_history'] = ray_history
        return outputs
    
    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        callbacks = []

        def set_step(step):
            self.step = step

        callbacks.append(
            TrainingCallback(
                where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                update_every_num_iters=1,
                func=set_step,
            )
        )
        return callbacks

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
            """Returns the parameter groups needed to optimizer your model components."""
            param_groups = {}
            param_groups["model"] = list(self.parameters())
            return param_groups
    

    def get_metrics_dict(self, outputs, batch):
        """Returns metrics dictionary which will be plotted with comet, wandb or tensorboard."""
        metrics_dict = {}
        gt_rgb = batch['image'].to(self.device)
        predicted_rgb = outputs['rgb']
        metrics_dict["psnr"] = self.psnr(gt_rgb, predicted_rgb)
        return metrics_dict
        
    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        """Returns a dictionary of losses to be summed which will be your loss."""
        loss_dict={}
        batch['lossmult'] = torch.Tensor([1.]).to(self.device)
        
        data_loss, stats = train_utils.compute_data_loss(batch, outputs['renderings'], self.zipnerf.config)
        loss_dict['data'] = data_loss
        
        if self.training:
            # interlevel loss in MipNeRF360
            # if self.config.interlevel_loss_mult > 0 and not self.config.single_mlp:
            #     loss_dict['interlevel'] = train_utils.interlevel_loss(outputs['ray_history'], self.config)

            # interlevel loss in ZipNeRF360
            if self.zipnerf.config.anti_interlevel_loss_mult > 0 and not self.zipnerf.single_mlp:
                loss_dict['anti_interlevel'] = train_utils.anti_interlevel_loss(outputs['ray_history'], self.zipnerf.config)

            # distortion loss
            if self.zipnerf.config.distortion_loss_mult > 0:
                loss_dict['distortion'] = train_utils.distortion_loss(outputs['ray_history'], self.zipnerf.config)

            # opacity loss
            # if self.config.opacity_loss_mult > 0:
            #     loss_dict['opacity'] = train_utils.opacity_loss(outputs['rgb'], self.config)

            # # orientation loss in RefNeRF
            # if (self.config.orientation_coarse_loss_mult > 0 or
            #         self.config.orientation_loss_mult > 0):
            #     loss_dict['orientation'] = train_utils.orientation_loss(batch, self.config, outputs['ray_history'],
            #                                                             self.config)
            # hash grid l2 weight decay
            if self.zipnerf.config.hash_decay_mults > 0:
                loss_dict['hash_decay'] = train_utils.hash_decay_loss(outputs['ray_history'], self.zipnerf.config)

            # # normal supervision loss in RefNeRF
            # if (self.config.predicted_normal_coarse_loss_mult > 0 or
            #         self.config.predicted_normal_loss_mult > 0):
            #     loss_dict['predicted_normals'] = train_utils.predicted_normal_loss(
            #         self.config, outputs['ray_history'], self.config)
        return loss_dict


    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]: # type: ignore
        """Returns a dictionary of images and metrics to plot. Here you can apply your colormaps."""
        gt_rgb = batch["image"].to(self.device)

        predicted_rgb = outputs["rgb"]
        # print('min,max:',predicted_rgb.min(),predicted_rgb.max())
        acc = colormaps.apply_colormap(outputs["accumulation"])
        depth = colormaps.apply_depth_colormap(outputs["depth"], accumulation=outputs["accumulation"])

        combined_rgb = torch.cat([gt_rgb, predicted_rgb], dim=1)
        combined_acc = torch.cat([acc], dim=1)
        combined_depth = torch.cat([depth], dim=1)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        gt_rgb = torch.moveaxis(gt_rgb, -1, 0)[None, ...]
        predicted_rgb = torch.moveaxis(predicted_rgb, -1, 0)[None, ...]

        # all of these metrics will be logged as scalars
        metrics_dict = {
            "psnr": float(self.psnr(gt_rgb, predicted_rgb).item()),
            "ssim": float(self.ssim(gt_rgb, torch.clip(predicted_rgb, 0.0, 1.0))),
            "lpips": float(self.lpips(gt_rgb, torch.clip(predicted_rgb, 0.0, 1.0)))
        }
        images_dict = {"img": combined_rgb, "accumulation": combined_acc, "depth": combined_depth}

        return metrics_dict, images_dict
    
"zipnerf_pipeline.py"

"""
Nerfstudio ZipNerf Pipeline
"""

import typing
from dataclasses import dataclass, field
from typing import Literal, Optional, Type

import torch.distributed as dist
from torch.cuda.amp.grad_scaler import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP

from zipnerf_ns.zipnerf_datamanager import ZipNerfDataManagerConfig
from zipnerf_ns.zipnerf_model import ZipNerfModel, ZipNerfModelConfig
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
class ZipNerfPipelineConfig(VanillaPipelineConfig):
    """Configuration for pipeline instantiation"""

    _target: Type = field(default_factory=lambda: ZipNerfPipeline)
    """target class to instantiate"""
    datamanager: DataManagerConfig = ZipNerfDataManagerConfig()
    """specifies the datamanager config"""
    model: ModelConfig = ZipNerfModelConfig()
    """specifies the model config"""


class ZipNerfPipeline(VanillaPipeline):
    """ZipNerf Pipeline

    Args:
        config: the pipeline config used to instantiate class
    """

    def __init__(
        self,
        config: ZipNerfPipelineConfig,
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
                ZipNerfModel, DDP(self._model, device_ids=[local_rank], find_unused_parameters=True)
            )
            dist.barrier(device_ids=[local_rank])