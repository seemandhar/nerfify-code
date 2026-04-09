"""
AR-NeRF Method Specification
Register AR-NeRF as a nerfstudio plugin method.
"""

from __future__ import annotations

from nerfify.methods.ar_nerf.ar_nerf_datamanager import ARNeRFDataManagerConfig
from nerfify.methods.ar_nerf.ar_nerf_model import ARNeRFModelConfig
from nerfify.methods.ar_nerf.ar_nerf_pipeline import ARNeRFPipelineConfig

from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.optimizers import RAdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.plugins.types import MethodSpecification


ar_nerf = MethodSpecification(
    config=TrainerConfig(
        method_name="ar_nerf",
        # Eval every 500 steps; save every 2000 steps; 100k total iterations.
        steps_per_eval_batch=500,
        steps_per_eval_image=500,
        steps_per_save=2000,
        steps_per_eval_all_images=1000000,
        # max_num_iterations=69768,
        mixed_precision=False,
        pipeline=ARNeRFPipelineConfig(
            datamanager=ARNeRFDataManagerConfig(
                # --- Dataset paths (override on CLI) ---
                # data_root="data/dtu",
                # ann_file_train="data/dtu/train.txt",
                # ann_file_eval="data/dtu/val.txt",
                # pairs_path="data/dtu/pairs.th",
                # 3-view setting by default; change to 6 or 9 as needed.
                # input_views=3,
                # DTU near/far bounds (paper uses 425–905 mm).
                # depth_ranges=(425.0, 905.0),
            ),
            model=ARNeRFModelConfig(
                # Sampling
                num_coarse_samples=64,
                num_importance_samples=128,
                # Frequency regularization: ramp up over first 50k steps.
                freq_reg_end=62791,
                num_freq_bands=16,
                # Two-phase blur: Ts = 10000 for 3-view DTU (paper Table impl.)
                blur_phase_end=10000,
                gaussian_kernel_size=3,
                # Loss weights (paper: lambda_u = lambda_o = 0.01)
                lambda_u=0.01,
                lambda_r_init=1e-5,
                lambda_r_final=1e-3,
                lambda_r_warmup_steps=512,
                lambda_o=0.01,
                ray_density_s=10.0,
                background_color="white",
                eval_num_rays_per_chunk=4096,
            ),
        ),
        optimizers={
            "fields": {
                # Paper uses Adam with exponential decay + warm-up.
                # "optimizer": AdamOptimizerConfig(lr=5e-4, eps=1e-8),
                "optimizer": RAdamOptimizerConfig(lr=2e-3, eps=1e-8),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=2e-5,
                    max_steps=100000,
                ),
            },
        },
        # viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        # vis="viewer",
    ),
    description=(
        "AR-NeRF: Few-shot NeRF by Adaptive Rendering Loss Regularization. "
        "Combines frequency regularization of positional encoding with "
        "two-phase Gaussian-blur supervision and uncertainty-based adaptive "
        "rendering loss weights to improve novel view synthesis from sparse inputs."
    ),
)