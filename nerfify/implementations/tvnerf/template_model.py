"""
TVNeRF Model — Nerfacto backbone + ray-TV and opacity regularizers
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Literal, Type

import torch
from torch import Tensor

# from nerfstudio.models.nerfacto import NerfactoModel, NerfactoModelConfig
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.models.vanilla_nerf import NeRFModel, VanillaModelConfig



@dataclass
class TvnerfModelConfig(VanillaModelConfig):
    """Config for TvnerfModel (inherits all Nerfacto knobs)."""
    _target: Type = field(default_factory=lambda: TvnerfModel)

    # TVNeRF regularization weights
    tv_lambda: float = 0.001
    """lambda_1: multiplier for TV noise loss."""
    opacity_lambda: float = 0.001
    """lambda_2: multiplier for opacity loss."""

    opacity_loss_type: Literal["alpha", "weight"] = "weight"
    """Use 'alpha' sum or 'weight' sum for opacity regularizer."""

    ray_hit_epsilon: float = 0.1
    """Threshold on accumulated opacity (sum of weights) to consider a ray as a hit."""

    tv_rho: float = 1.0
    """Sigmoid bias term for TV noise."""
    tv_omega: float = -1.0
    """Sigmoid slope term for TV noise (should be negative)."""

    use_mask_on_opacity_loss: bool = True
    """If True, apply opacity loss to non-hitting unseen rays only (recommended)."""


class TvnerfModel(NeRFModel):
    """Nerfacto-based model with TVNeRF regularizers."""

    config: TvnerfModelConfig

    def populate_modules(self) -> None:
        """Reuse Nerfacto composition (proposal nets, field, renderers, losses, metrics)."""
        super().populate_modules()

    @staticmethod
    def _alphas_from_weights(weights: Tensor, eps: float = 1e-8) -> Tensor:
        """Reconstruct alpha sequence from volumetric weights.

        Args:
            weights: [num_rays, num_samples, 1]
        Returns:
            alphas: [num_rays, num_samples, 1]
        """
        assert weights.ndim >= 3 and weights.shape[-1] == 1, "weights must be [..., S, 1]"
        B, S = weights.shape[0], weights.shape[1]
        device = weights.device
        T = torch.ones((B, 1, 1), device=device, dtype=weights.dtype)
        alphas_list = []
        for i in range(S):
            wi = weights[:, i : i + 1, :]  # [B,1,1]
            ai = wi / (T + eps)
            ai = ai.clamp(min=0.0, max=1.0)
            alphas_list.append(ai)
            T = T * (1.0 - ai)
        alphas = torch.cat(alphas_list, dim=1)
        return alphas

    def _ray_tv_noise_loss(self, weights: Tensor) -> Tensor:
        """Compute per-ray TV noise from weights via normalized alphas.

        Args:
            weights: [num_rays, num_samples, 1]
        Returns:
            tv_noise_per_ray: [num_rays]
        """
        alphas = self._alphas_from_weights(weights)  # [B,S,1]
        denom = alphas.sum(dim=-2, keepdim=True) + 1e-8  # [B,1,1]
        p = alphas / denom
        dp = torch.abs(p[:, 1:, :] - p[:, :-1, :]).sum(dim=-2)  # [B,1]
        tv = dp.squeeze(-1)  # [B]
        tv_noise = torch.sigmoid(self.config.tv_rho + self.config.tv_omega * tv)
        return tv_noise  # [B]

    def _opacity_loss(self, weights: Tensor, loss_type: str) -> Tensor:
        """Compute per-ray opacity loss from weights or alphas.

        Args:
            weights: [num_rays, num_samples, 1]
            loss_type: 'alpha' or 'weight'
        Returns:
            per-ray opacity quantity [num_rays]
        """
        if loss_type == "weight":
            x = weights.sum(dim=-2).squeeze(-1)  # [B]
        elif loss_type == "alpha":
            alphas = self._alphas_from_weights(weights)
            x = alphas.sum(dim=-2).squeeze(-1)  # [B]
        else:
            raise ValueError(f"Unknown opacity_loss_type: {loss_type}")
        return x  # [B]

    def get_outputs(self, ray_bundle: RayBundle):
        if self.field_coarse is None or self.field_fine is None:
            raise ValueError("populate_fields() must be called before get_outputs")

        # uniform sampling
        ray_samples_uniform = self.sampler_uniform(ray_bundle)
        if self.temporal_distortion is not None:
            offsets = None
            if ray_samples_uniform.times is not None:
                offsets = self.temporal_distortion(
                    ray_samples_uniform.frustums.get_positions(), ray_samples_uniform.times
                )
            ray_samples_uniform.frustums.set_offsets(offsets)

        # coarse field:
        field_outputs_coarse = self.field_coarse.forward(ray_samples_uniform)
        if self.config.use_gradient_scaling:
            field_outputs_coarse = scale_gradients_by_distance_squared(field_outputs_coarse, ray_samples_uniform)
        weights_coarse = ray_samples_uniform.get_weights(field_outputs_coarse[FieldHeadNames.DENSITY])
        rgb_coarse = self.renderer_rgb(
            rgb=field_outputs_coarse[FieldHeadNames.RGB],
            weights=weights_coarse,
        )
        accumulation_coarse = self.renderer_accumulation(weights_coarse)
        depth_coarse = self.renderer_depth(weights_coarse, ray_samples_uniform)

        # pdf sampling
        ray_samples_pdf = self.sampler_pdf(ray_bundle, ray_samples_uniform, weights_coarse)
        if self.temporal_distortion is not None:
            offsets = None
            if ray_samples_pdf.times is not None:
                offsets = self.temporal_distortion(ray_samples_pdf.frustums.get_positions(), ray_samples_pdf.times)
            ray_samples_pdf.frustums.set_offsets(offsets)

        # fine field:
        field_outputs_fine = self.field_fine.forward(ray_samples_pdf)
        if self.config.use_gradient_scaling:
            field_outputs_fine = scale_gradients_by_distance_squared(field_outputs_fine, ray_samples_pdf)
        weights_fine = ray_samples_pdf.get_weights(field_outputs_fine[FieldHeadNames.DENSITY])
        rgb_fine = self.renderer_rgb(
            rgb=field_outputs_fine[FieldHeadNames.RGB],
            weights=weights_fine,
        )
        accumulation_fine = self.renderer_accumulation(weights_fine)
        depth_fine = self.renderer_depth(weights_fine, ray_samples_pdf)

        outputs = {
            "rgb_coarse": rgb_coarse,
            "rgb_fine": rgb_fine,
            "accumulation_coarse": accumulation_coarse,
            "accumulation_fine": accumulation_fine,
            "depth_coarse": depth_coarse,
            "depth_fine": depth_fine,
        }
        if(self.training):
            # outputs['weights'] = weights_fine
            outputs['weights'] = weights_coarse

        return outputs

    
    def get_loss_dict(self, outputs: Dict, batch: Dict, metrics_dict=None) -> Dict:
        """Start with Nerfacto losses; add TVNeRF regularizers using unseen rays."""
        loss_dict = super().get_loss_dict(outputs, batch, metrics_dict)

        unseen_rb = batch["unseen_ray_bundle"]

        unseen_outputs = self.get_outputs(unseen_rb)

        weights_unseen: Tensor = unseen_outputs["weights"]  # [B,S,1]
        accumulation_unseen: Tensor = unseen_outputs["accumulation_coarse"].squeeze(-1)  # [B]

        # Ray-hit mask based on accumulated opacity (sum of weights)
        hit_mask = (accumulation_unseen > self.config.ray_hit_epsilon).float()  # [B]
        non_hit_mask = 1.0 - hit_mask

        # TV noise on (predicted) hitting unseen rays
        if self.config.tv_lambda > 0:
            tv_noise = self._ray_tv_noise_loss(weights_unseen)  # [B]
            # average over hits to keep scale consistent
            denom_hits = torch.clamp(hit_mask.sum(), min=1.0)
            tv_loss = (tv_noise * hit_mask).sum() / hit_mask.shape[0]
            loss_dict["tv_loss"] = self.config.tv_lambda * tv_loss

        # Opacity loss on non-hitting unseen rays (reduce floaters)
        if self.config.opacity_lambda > 0:
            opacity_per_ray = self._opacity_loss(weights_unseen, self.config.opacity_loss_type)  # [B]
            if self.config.use_mask_on_opacity_loss:
                denom_non_hits = torch.clamp(non_hit_mask.sum(), min=1.0)
                opacity_loss = (opacity_per_ray * non_hit_mask).sum() / hit_mask.shape[0]
            else:
                opacity_loss = opacity_per_ray.mean()
            loss_dict["opacity_loss"] = self.config.opacity_lambda * opacity_loss

        return loss_dict


    def get_loss_dict_(self, outputs: Dict, batch: Dict, metrics_dict=None) -> Dict:
        """Start with Nerfacto losses; add TVNeRF regularizers using unseen rays."""
        loss_dict = super().get_loss_dict(outputs, batch, metrics_dict)
        if(not self.training):
            return loss_dict

        # Retrieve final-stage weights and accumulation
        assert "weights" in outputs
        weights_unseen = outputs["weights"]  # [B,S,1]
        accumulation = outputs["accumulation_coarse"].squeeze(-1)  # [B]

        # Ray-hit mask based on accumulated opacity (sum of weights)
        hit_mask = (accumulation > self.config.ray_hit_epsilon).float()  # [B]
        non_hit_mask = 1.0 - hit_mask

        # TV noise on (predicted) hitting unseen rays
        if self.config.tv_lambda > 0:
            tv_noise = self._ray_tv_noise_loss(weights_unseen)  # [B]
            # average over hits to keep scale consistent
            denom_hits = torch.clamp(hit_mask.sum(), min=1.0)
            tv_loss = (tv_noise * hit_mask).sum() / denom_hits
            loss_dict["tv_loss"] = self.config.tv_lambda * tv_loss

        # Opacity loss on non-hitting unseen rays (reduce floaters)
        if self.config.opacity_lambda > 0:
            opacity_per_ray = self._opacity_loss(weights_unseen, self.config.opacity_loss_type)  # [B]
            if self.config.use_mask_on_opacity_loss:
                denom_non_hits = torch.clamp(non_hit_mask.sum(), min=1.0)
                opacity_loss = (opacity_per_ray * non_hit_mask).sum() / denom_non_hits
            else:
                opacity_loss = opacity_per_ray.mean()
            loss_dict["opacity_loss"] = self.config.opacity_lambda * opacity_loss

        return loss_dict
