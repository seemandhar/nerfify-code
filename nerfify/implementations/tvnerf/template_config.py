"""
TVNeRF Config — method wiring to Nerfacto backbone with TV/opacity regularizers.
"""
from __future__ import annotations
from nerfify.methods.tvnerf.template_datamanager import TvnerfDataManagerConfig
from nerfify.methods.tvnerf.template_model import TvnerfModelConfig
from nerfify.methods.tvnerf.template_pipeline import TvnerfPipelineConfig
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig, RAdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.plugins.types import MethodSpecification

tvnerf = MethodSpecification(
    config=TrainerConfig(
        method_name="tvnerf",
        steps_per_eval_batch=500,
        steps_per_save=2000,
        max_num_iterations=30000,
        mixed_precision=True,
        pipeline=TvnerfPipelineConfig(
            datamanager=TvnerfDataManagerConfig(
                dataparser=NerfstudioDataParserConfig(),
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=4096,
                unseen_num_rays_per_batch=1024,
            ),
            model=TvnerfModelConfig(
                eval_num_rays_per_chunk=1 << 15,
                # TVNeRF regularizer defaults from paper
                tv_lambda=0.1,
                opacity_lambda=0.1,
                opacity_loss_type="weight",  # Blender: "weight"; LLFF: "alpha"
                ray_hit_epsilon=0.1,  # LLFF recommended: 0.8
                tv_rho=1.0,
                tv_omega=-1.0,
            ),
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
        vis="viewer",
    ),
    description="TVNeRF: Nerfacto-based few-view regularization via ray TV-max and opacity-min.",
)
