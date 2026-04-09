"""
Nerfstudio Template Config — TARGET PAPER method wiring
"""
from __future__ import annotations

from nerfify.methods.hyb_nerf.template_datamanager import TemplateDataManagerConfig
from nerfify.methods.hyb_nerf.template_model import TemplateModelConfig
from nerfify.methods.hyb_nerf.template_pipeline import TemplatePipelineConfig
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig, RAdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.plugins.types import MethodSpecification

hyb_nerf = MethodSpecification(
    config=TrainerConfig(
        method_name="hyb-nerf",          # shows up in viewer/logs
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
                # Hyb-NeRF knobs (coarse Fourier + fine hash)
                coarse_num_frequencies=8,
                hash_num_levels=16,
                hash_features_per_level=2,
                hash_log2_hashmap_size=19,
                hash_min_res=180,
                hash_max_res=4096,
                density_hidden_dim=64,
                color_hidden_dim=64,
                geo_feat_dim=15,
                use_cone_embedding=True,
                # extra knob exposed for Hyb-NeRF hooks
                l2_appearance_mult=1e-4,
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
    description="Hyb-NeRF: Nerfacto-based method with hybrid (learned Fourier + hash) multiresolution encoding.",
)
