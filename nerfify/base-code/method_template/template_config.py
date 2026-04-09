"""
Template Config — Method registration and hyperparameter wiring.

This is the entry point for nerfstudio, discovered via pyproject.toml.
Optimizer defaults are proven across 6+ tested NeRF implementations.
"""
from __future__ import annotations

from .template_datamanager import TemplateDataManagerConfig
from .template_model import TemplateModelConfig
from .template_pipeline import TemplatePipelineConfig

from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig, RAdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.plugins.types import MethodSpecification


method_spec = MethodSpecification(
    config=TrainerConfig(
        method_name="my-method",
        steps_per_eval_batch=500,
        steps_per_save=2000,
        max_num_iterations=30000,
        mixed_precision=True,
        pipeline=TemplatePipelineConfig(
            datamanager=TemplateDataManagerConfig(
                dataparser=NerfstudioDataParserConfig(),
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=4096,
            ),
            model=TemplateModelConfig(
                eval_num_rays_per_chunk=1 << 15,
                hash_num_levels=16,
                hash_features_per_level=2,
                hash_log2_hashmap_size=19,
                hash_min_res=16,
                hash_max_res=2048,
                hidden_dim_density=64,
                hidden_dim_color=64,
                geo_feat_dim=15,
                l2_appearance_mult=0.0,
            ),
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=5e-4, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=0.0001, max_steps=200000
                ),
            },
            "fields": {
                "optimizer": RAdamOptimizerConfig(lr=5e-4, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1e-4, max_steps=50000
                ),
            },
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1e-4, max_steps=5000
                ),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="Template method — replace with your paper's method description.",
)
