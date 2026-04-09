from __future__ import annotations

from nerfify.methods.mi_nerf.minerf_datamanager import MiNeRFDataManagerConfig
from nerfify.methods.mi_nerf.minerf_model import MiNeRFModelConfig
from nerfify.methods.mi_nerf.minerf_pipeline import MiNeRFPipelineConfig
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig, RAdamOptimizerConfig
from nerfstudio.engine.schedulers import (
    ExponentialDecaySchedulerConfig,
)
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.plugins.types import MethodSpecification


minerf = MethodSpecification(
    config=TrainerConfig(
        method_name="minerf", 
        steps_per_eval_batch=100,
        steps_per_save=2000,
        max_num_iterations=26000,
        mixed_precision=False,
        pipeline=MiNeRFPipelineConfig(
            datamanager=MiNeRFDataManagerConfig(
                train_num_rays_per_batch=4096, #4096
                eval_num_rays_per_batch=4096, #4096
            ),
            model=MiNeRFModelConfig(
                eval_num_rays_per_chunk=4096,
            ),
        ),
        optimizers={
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
        # viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        # vis="wandb", #"viewer+wandb"
    ),
    description="Implementation for MiNeRF.",
)
