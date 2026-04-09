from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple, Type

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.field_components.encodings import NeRFEncoding
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.model_components.losses import MSELoss
from nerfstudio.model_components.ray_samplers import PDFSampler, UniformSampler
from nerfstudio.model_components.renderers import (
    AccumulationRenderer,
    DepthRenderer,
    RGBRenderer,
)
from nerfstudio.fields.vanilla_nerf_field import NeRFField
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.models.vanilla_nerf import NeRFModel, VanillaModelConfig
from nerfstudio.utils import colormaps

from nerfify.methods.ar_nerf.ar_nerf_field import ARNeRFField


def get_freq_mask(t: int, T: int, L: int, pe_dim: int, device) -> Tensor:
    mask = torch.zeros(pe_dim, device=device)
    ratio = t * L / max(T, 1)
    # ratio = int(0.1 * L * 2) 
    for i in range(pe_dim):
        if i <= ratio + 3:
            mask[i] = 1.0
        elif ratio + 3 < i <= ratio + 6:
            mask[i] = ratio - int(ratio)
        else:
            mask[i] = 0.0
    return mask

def gaussian_blur_batch(images: Tensor, kernel_size: int = 3) -> Tensor:
    squeeze = images.dim() == 3
    if squeeze:
        images = images.unsqueeze(0)

    B, H, W, C = images.shape
    imgs = images.permute(0, 3, 1, 2)  # (B, C, H, W)

    # Build simple Gaussian kernel
    sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8
    ax = torch.arange(kernel_size, device=images.device, dtype=torch.float32)
    ax = ax - kernel_size // 2
    kernel_1d = torch.exp(-0.5 * (ax / sigma) ** 2)
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]  # (k, k)
    kernel_2d = kernel_2d.view(1, 1, kernel_size, kernel_size).expand(C, 1, -1, -1)

    pad = kernel_size // 2
    blurred = F.conv2d(imgs, kernel_2d, padding=pad, groups=C)
    blurred = blurred.permute(0, 2, 3, 1)  # (B, H, W, C)

    if squeeze:
        blurred = blurred.squeeze(0)
    return blurred



def ray_density_regularization(alphas: Tensor, s: float = 10.0) -> Tensor:
    alpha_sum = alphas.sum(dim=-1, keepdim=True).clamp(min=1e-6)
    p = alphas / alpha_sum  # (N_rays, N_samples) — ray density

    loss = torch.log(1.0 + s * p).mean()
    return loss


from nerfstudio.configs.config_utils import to_immutable_dict
@dataclass
class ARNeRFModelConfig(VanillaModelConfig):
    """AR-NeRF Model Configuration."""

    _target: Type = field(default_factory=lambda: ARNeRFModel)

    # collider_params: Optional[Dict[str, float]] = to_immutable_dict({"near_plane": 0.0, "far_plane": 1.0})


    # Sampling
    num_coarse_samples: int = 64
    """Number of coarse samples along each ray."""
    num_importance_samples: int = 128
    """Number of fine (PDF) samples along each ray."""

    # Frequency regularization (Sec. 3.2 / Eq. 5–6)
    freq_reg_end: int = 62791
    """Iteration T at which all PE frequencies are fully enabled."""
    num_freq_bands: int = 16
    """Number of frequency bands L in positional encoding (same as NeRF)."""

    # Two-phase rendering supervision (Sec. 3.2 / Eq. 7)
    blur_phase_end: int = 10000
    """Iteration Ts after which raw (unblurred) images are used. Paper sets
    10000 / 15000 / 16000 for 3 / 6 / 9 views on DTU."""
    gaussian_kernel_size: int = 3
    """Gaussian blur kernel size (paper uses 3)."""

    # Loss weights (Eq. 15)
    lambda_u: float = 0.01
    """Weight for adaptive uncertainty loss L_u."""
    lambda_r_init: float = 1e-5
    """Initial weight for ray density regularization L_r."""
    lambda_r_final: float = 1e-3
    """Final weight for L_r (reached at lambda_r_warmup_steps)."""
    lambda_r_warmup_steps: int = 512
    """Steps over which lambda_r is linearly warmed up."""
    lambda_o: float = 0.01
    """Weight for occlusion regularization L_o."""
    ray_density_s: float = 10.0
    """Steepness constant s for ray density regularization."""

    background_color: Literal["random", "last_sample", "black", "white"] = "white"


class ARNeRFModel(NeRFModel):

    config: ARNeRFModelConfig

    def __init__(self, config: ARNeRFModelConfig, **kwargs) -> None:
        self.field_coarse = None
        self.field_fine = None
        self.temporal_distortion = None
        self._step = 0  # updated externally by the pipeline
        super().__init__(config=config, **kwargs)

    def populate_modules(self):
        """Set up fields and modules."""
        super().populate_modules()

        L = self.config.num_freq_bands  # 10 by default

        position_encoding = NeRFEncoding(
            in_dim=3,
            num_frequencies=L,
            min_freq_exp=0.0,
            max_freq_exp=L,
            include_input=True,
        )
        direction_encoding = NeRFEncoding(
            in_dim=3,
            num_frequencies=4,
            min_freq_exp=0.0,
            max_freq_exp=4.0,
            include_input=True,
        )

        # self.field_coarse = ARNeRFField(
        self.field_coarse = NeRFField(
            position_encoding=position_encoding,
            direction_encoding=direction_encoding,
            use_integrated_encoding=True
        )
        # self.field_fine = ARNeRFField(
        self.field_fine = NeRFField(
            position_encoding=position_encoding,
            direction_encoding=direction_encoding,
            use_integrated_encoding=True
        )

        # Cache PE output dim for frequency mask
        self._pe_dim = position_encoding.get_out_dim()

    def _apply_freq_mask(self, ray_samples, field):
        t = self._step
        T = self.config.freq_reg_end
        L = self.config.num_freq_bands
        device = next(self.parameters()).device

        mask = get_freq_mask(t, T, L, self._pe_dim, device)  # (pe_dim,)

        # Monkey-patch the position encoding's forward to apply the mask.
        pos_enc = field.position_encoding
        original_forward = pos_enc.forward

        def masked_forward(in_tensor, covs=None):
            out = original_forward(in_tensor) if covs is None else original_forward(in_tensor, covs=covs)
            return out * mask
            # return out

        pos_enc.forward = masked_forward
        field_outputs = field.forward(ray_samples)
        pos_enc.forward = original_forward  # restore
        return field_outputs

    def _get_pixel_supervision(self, gt_pixels: Tensor) -> Tensor:
        if self._step < self.config.blur_phase_end:
            # Reshape to tiny pseudo-image for blurring, then flatten back.
            # For ray-based training we blur per-ray by treating it as a 1D strip.
            N = gt_pixels.shape[0]
            W = max(1, int(N ** 0.5))
            H = (N + W - 1) // W
            pad = H * W - N
            padded = torch.cat([gt_pixels, gt_pixels[:pad]], dim=0) if pad > 0 else gt_pixels
            img = padded.view(1, H, W, 3)
            blurred = gaussian_blur_batch(img, self.config.gaussian_kernel_size)
            blurred = blurred.view(H * W, 3)[:N]
            return blurred.detach()
        else:
            return gt_pixels
        
    def _get_lambda_r(self) -> float:
        t = self._step
        ws = self.config.lambda_r_warmup_steps
        if t >= ws:
            return self.config.lambda_r_final
        alpha = t / max(ws, 1)
        return self.config.lambda_r_init + alpha * (self.config.lambda_r_final - self.config.lambda_r_init)

    def get_outputs(self, ray_bundle: RayBundle) -> Dict[str, Tensor]:
        """Run coarse + fine NeRF with frequency-masked PE."""
        if self.field_coarse is None or self.field_fine is None:
            raise ValueError("populate_modules() must be called first.")

        # Coarse sampling
        ray_samples_uniform = self.sampler_uniform(ray_bundle)

        field_outputs_coarse = self._apply_freq_mask(ray_samples_uniform, self.field_coarse)

        weights_coarse = ray_samples_uniform.get_weights(
            field_outputs_coarse[FieldHeadNames.DENSITY]
        )
        rgb_coarse = self.renderer_rgb(
            rgb=field_outputs_coarse[FieldHeadNames.RGB], weights=weights_coarse
        )
        accumulation_coarse = self.renderer_accumulation(weights_coarse)
        depth_coarse = self.renderer_depth(weights_coarse, ray_samples_uniform)

        # Rendered variance for coarse (Eq. 10): beta_bar^2 = sum_i w_i^2 * beta_i^2
        # beta2_coarse_pts = field_outputs_coarse.get("beta2", None)  # (N_pts, N_samples, 1)

        # beta2_coarse_pts = None

        # if beta2_coarse_pts is not None:
        #     w2_coarse = weights_coarse ** 2  # (N_rays, N_samples, 1)
        #     # beta2_coarse = (w2_coarse * beta2_coarse_pts.squeeze(-1).unsqueeze(0)).sum(dim=-1)
        #     beta2_coarse = (w2_coarse * beta2_coarse_pts)[..., 0].sum(dim=-1)  # (N_rays,)
        # else:
        #     beta2_coarse = None

        # Fine (PDF) sampling
        ray_samples_pdf = self.sampler_pdf(ray_bundle, ray_samples_uniform, weights_coarse)
        field_outputs_fine = self._apply_freq_mask(ray_samples_pdf, self.field_fine)

        weights_fine = ray_samples_pdf.get_weights(field_outputs_fine[FieldHeadNames.DENSITY])
        rgb_fine = self.renderer_rgb(
            rgb=field_outputs_fine[FieldHeadNames.RGB], weights=weights_fine
        )
        accumulation_fine = self.renderer_accumulation(weights_fine)
        depth_fine = self.renderer_depth(weights_fine, ray_samples_pdf)

        # beta2_fine_pts = field_outputs_fine.get("beta2", None)
        # beta2_fine_pts = None
        # if beta2_fine_pts is not None:
        #     w2_fine = weights_fine ** 2
        #     # beta2_fine = (w2_fine * beta2_fine_pts.squeeze(-1).unsqueeze(0)).sum(dim=-1)
        #     beta2_fine = (w2_fine * beta2_fine_pts)[..., 0].sum(dim=-1)  # (N_rays,)
        # else:
        #     beta2_fine = None

        outputs = {
            "rgb_coarse": rgb_coarse,
            "rgb_fine": rgb_fine,
            "accumulation_coarse": accumulation_coarse,
            "accumulation_fine": accumulation_fine,
            "depth_coarse": depth_coarse,
            "depth_fine": depth_fine,
            # Alpha values for ray density regularization
            "alphas_coarse": field_outputs_coarse[FieldHeadNames.DENSITY].squeeze(-1),
            "alphas_fine": field_outputs_fine[FieldHeadNames.DENSITY].squeeze(-1),
            # Weights for occlusion regularization
            "weights_fine": weights_fine,
        }
        # if beta2_coarse is not None:
        #     outputs["beta2_coarse"] = beta2_coarse
        # if beta2_fine is not None:
        #     outputs["beta2_fine"] = beta2_fine

        return outputs

    def get_loss_dict(
        self,
        outputs: Dict[str, Tensor],
        batch: Dict[str, Any],
        metrics_dict: Optional[Dict] = None,
    ) -> Dict[str, Tensor]:
        """Compute total loss (Eq. 15).

        L_total = L_s + lambda_u * L_u + lambda_r * L_r + lambda_o * L_o
        """
        device = outputs["rgb_coarse"].device

        # Ground-truth image pixels
        image = batch["image"].to(device)  # (N_rays, 3)


        coarse_pred, coarse_sup = self.renderer_rgb.blend_background_for_loss_computation(
            pred_image=outputs["rgb_coarse"],
            pred_accumulation=outputs["accumulation_coarse"],
            gt_image=image,
            # gt_image=supervision,
        )
        fine_pred, fine_sup = self.renderer_rgb.blend_background_for_loss_computation(
            pred_image=outputs["rgb_fine"],
            pred_accumulation=outputs["accumulation_fine"],
            gt_image=image,
            # gt_image=supervision,
        )

        L_s_coarse = self.rgb_loss(coarse_sup, coarse_pred)
        L_s_fine = self.rgb_loss(fine_sup, fine_pred)
        L_s = L_s_coarse + L_s_fine

        loss_dict: Dict[str, Tensor] = {"L_s": L_s}

        if "beta2_fine" in outputs:
            beta2 = outputs["beta2_fine"].clamp(min=1e-6)  # (N_rays,)
            _, raw_sup = self.renderer_rgb.blend_background_for_loss_computation(
                pred_image=outputs["rgb_fine"],
                pred_accumulation=outputs["accumulation_fine"],
                gt_image=image,
            )
            sq_err = ((raw_sup - fine_pred) ** 2).mean(dim=-1)  # (N_rays,)
            L_u = (sq_err / (2.0 * beta2) + 0.5 * torch.log(beta2)).mean()
            loss_dict["L_u"] = self.config.lambda_u * L_u

        if "alphas_fine" in outputs:
            alphas = outputs["alphas_fine"]  # (N_pts,) — flatten as 1-D batch
            if alphas.dim() == 1:
                alphas = alphas.unsqueeze(0)
            L_r = ray_density_regularization(alphas, s=self.config.ray_density_s)
            loss_dict["L_r"] = self._get_lambda_r() * L_r

        if "weights_fine" in outputs:
            weights = outputs["weights_fine"]  # (N_rays, N_samples, 1)
            if weights.dim() == 3:
                weights = weights.squeeze(-1)
            # Penalise weights for samples closest to camera (first ~10%)
            n_near = max(1, weights.shape[-1] // 10)
            L_o = weights[:, :n_near].mean()
            loss_dict["L_o"] = self.config.lambda_o * L_o

        return loss_dict

    def get_image_metrics_and_images(
        self,
        outputs: Dict[str, Tensor],
        batch: Dict[str, Any],
    ) -> Tuple[Dict[str, float], Dict[str, Tensor]]:
        image = batch["image"].to(outputs["rgb_coarse"].device)
        image = self.renderer_rgb.blend_background(image)
        rgb_coarse = outputs["rgb_coarse"]
        rgb_fine = outputs["rgb_fine"]
        acc_coarse = colormaps.apply_colormap(outputs["accumulation_coarse"])
        acc_fine = colormaps.apply_colormap(outputs["accumulation_fine"])

        combined_rgb = torch.cat([image, rgb_coarse, rgb_fine], dim=1)
        combined_acc = torch.cat([acc_coarse, acc_fine], dim=1)

        image_t = torch.moveaxis(image, -1, 0)[None, ...]
        rgb_coarse_t = torch.moveaxis(rgb_coarse, -1, 0)[None, ...]
        rgb_fine_t = torch.moveaxis(rgb_fine, -1, 0)[None, ...]

        coarse_psnr = self.psnr(image_t, rgb_coarse_t)
        fine_psnr = self.psnr(image_t, rgb_fine_t)
        # H = outputs['H']
        # W = outputs['W']
        fine_ssim = self.ssim(image_t, rgb_fine_t)
        # fine_ssim = self.ssim(image_t.reshape(image_t.shape[0], 3, H, W), rgb_fine_t.reshape(rgb_fine_t.shape[0], 3, H, W))
        fine_lpips = self.lpips(image_t, rgb_fine_t)
        # fine_lpips = self.lpips(image_t.reshape(image_t.shape[0], 3, H, W), rgb_fine_t.reshape(rgb_fine_t.shape[0], 3, H, W))

        metrics_dict = {
            "psnr": float(fine_psnr.item()),
            "coarse_psnr": float(coarse_psnr.item()),
            "fine_psnr": float(fine_psnr.item()),
            "fine_ssim": float(fine_ssim.item()),
            "fine_lpips": float(fine_lpips.item()),
        }
        print(coarse_psnr.item(), fine_psnr.item(), fine_ssim.item(), fine_lpips.item())
        images_dict = {
            "img": combined_rgb,
            "accumulation": combined_acc,
        }
        return metrics_dict, images_dict

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups: Dict[str, List[Parameter]] = {}
        if self.field_coarse is None or self.field_fine is None:
            raise ValueError("populate_modules() must be called first.")
        param_groups["fields"] = list(self.field_coarse.parameters()) + list(
            self.field_fine.parameters()
        )
        return param_groups