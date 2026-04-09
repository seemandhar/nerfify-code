"""
Nerfstudio Config — NeRF in detail (learning to sample)
"""
from __future__ import annotations

from nerfify.methods.nerf_id.template_datamanager import TemplateDataManagerConfig
from nerfify.methods.nerf_id.template_model import TemplateModelConfig
from nerfify.methods.nerf_id.template_pipeline import TemplatePipelineConfig
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.plugins.types import MethodSpecification

nerf_id = MethodSpecification(
    config=TrainerConfig(
        method_name="nerf_id",
        steps_per_eval_batch=500,
        steps_per_save=2000,
        max_num_iterations=200_000,
        mixed_precision=True,
        pipeline=TemplatePipelineConfig(
            datamanager=TemplateDataManagerConfig(
                dataparser=NerfstudioDataParserConfig(),
                train_num_rays_per_batch=4096,  # ~66k in paper, 65,536 practical
                eval_num_rays_per_batch=4096,
            ),
            model=TemplateModelConfig(
                eval_num_rays_per_chunk=1 << 15,
                # proposer defaults from TARGET PAPER
                num_coarse_samples=64,
                num_fine_samples=128,
                proposer_type="mlpmix",
                stage_mimic_steps=50_000,
                mimic_loss_mult=1.0,
                coarse_loss_mult=1.0,
                importance_loss_mult=1.0,
                importance_threshold=0.03,
                l2_appearance_mult=1e-4,
            ),
        ),
        optimizers={
            # Keep stable defaults used broadly across Nerfstudio methods
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=200000),
            },
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=5e-4, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=5e-5, max_steps=30000, warmup_steps=1000),
            },
            "proposer": {
                "optimizer": AdamOptimizerConfig(lr=5e-4, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=5e-5, max_steps=30000, warmup_steps=1000),
            },
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=5000),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="NeRF in detail: replace non-trainable hierarchical sampling with a learnable proposer trained in two stages.",
)
