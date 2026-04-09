"""
Nerfstudio Template Config — TARGET PAPER method wiring
"""
from __future__ import annotations

from nerfify.methods.surface_sampling.template_datamanager import TemplateDataManagerConfig
from nerfify.methods.surface_sampling.template_model import TemplateModelConfig
from nerfify.methods.surface_sampling.template_pipeline import TemplatePipelineConfig
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig, RAdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.plugins.types import MethodSpecification

surface_sampling = MethodSpecification(
    config=TrainerConfig(
        method_name="surface_sampling",          # shows up in viewer/logs
        steps_per_eval_batch=500,
        steps_per_save=2000,
        max_num_iterations=400000,               # per paper
        mixed_precision=True,
        pipeline=TemplatePipelineConfig(
            datamanager=TemplateDataManagerConfig(
                dataparser=NerfstudioDataParserConfig(),
                train_num_rays_per_batch=4096,   # per paper
                eval_num_rays_per_batch=4096,
                enable_near_surface_sampling=True,
                near_surface_alpha=1.0 / 8.0,    # default synthetic
                ns_num_samples=64,
                jitter_near_surface=True,
                use_pointcloud_for_eval=True,
                pc_offline_num_views=20,
                pc_stride=4,
                pc_tau=0.1,
                depth_hole_fill_kappa=2.0,
                depth_hole_fill_window=11,
            ),
            model=TemplateModelConfig(
                eval_num_rays_per_chunk=1 << 15,
                # extra knob exposed for TARGET PAPER hooks
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
    description="TARGET PAPER: Nerfacto-based method with near-surface sampling and point-cloud-derived test-time depth.",
)
