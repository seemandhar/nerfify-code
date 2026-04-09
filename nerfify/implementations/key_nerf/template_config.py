"""
Nerfstudio Template Config — Informative Rays Selection for Few-Shot NeRFs
"""
from __future__ import annotations

from nerfify.methods.key_nerf.template_datamanager import TemplateDataManagerConfig
from nerfify.methods.key_nerf.template_model import TemplateModelConfig
from nerfify.methods.key_nerf.template_pipeline import TemplatePipelineConfig
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig, RAdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.plugins.types import MethodSpecification

keynerf = MethodSpecification(
    config=TrainerConfig(
        method_name="keynerf",  # shows up in viewer/logs
        steps_per_eval_batch=500,
        steps_per_save=2000,
        max_num_iterations=50000,
        mixed_precision=True,
        pipeline=TemplatePipelineConfig(
            datamanager=TemplateDataManagerConfig(
                dataparser=NerfstudioDataParserConfig(),
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=4096,
                num_selected_cameras=16,
                entropy_mix_frac=0.5,
                entropy_window=9,
                entropy_use_gradient=True,
                use_view_selection=True,
            ),
            model=TemplateModelConfig(
                eval_num_rays_per_chunk=1 << 15,
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
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="Informative Rays Selection for Few-Shot NeRFs: view subset + entropy-biased rays on top of Nerfacto.",
)
