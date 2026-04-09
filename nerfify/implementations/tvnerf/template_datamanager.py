"""
TVNeRF DataManager

Extends VanillaDataManager to also sample a batch of 'unseen' rays (from eval split if available)
for TV/opacity regularization that does not require RGB supervision.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Literal, Tuple, Type, Union

import torch
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.datamanagers.base_datamanager import (
    VanillaDataManager,
    VanillaDataManagerConfig,
)


@dataclass
class TvnerfDataManagerConfig(VanillaDataManagerConfig):
    """TVNeRF DataManager Config."""
    unseen_num_rays_per_batch: int = 1024
    """Number of unseen rays to sample per training iteration for regularizers."""
    _target: Type = field(default_factory=lambda: TvnerfDataManager)


class TvnerfDataManager(VanillaDataManager):
    """Thin wrapper around VanillaDataManager with unseen-ray sampling."""
    config: TvnerfDataManagerConfig

    def __init__(
        self,
        config: TvnerfDataManagerConfig,
        device: Union[torch.device, str] = "cpu",
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        **kwargs,
    ):
        super().__init__(
            config=config, device=device, test_mode=test_mode, world_size=world_size, local_rank=local_rank, **kwargs
        )

    def _slice_batch(self, batch: Dict, num: int) -> Dict:
        """Utility to slice a pixel-sampled batch to the first `num` rays."""
        out: Dict = {}
        for k, v in batch.items():
            if torch.is_tensor(v) and v.shape[0] >= num:
                out[k] = v[:num]
            else:
                out[k] = v
        return out

    def next_train(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the train dataloader plus unseen-ray bundle."""
        self.train_count += 1
        image_batch = next(self.iter_train_image_dataloader)
        assert self.train_pixel_sampler is not None
        assert isinstance(image_batch, dict)
        batch = self.train_pixel_sampler.sample(image_batch)
        ray_indices = batch["indices"]
        ray_bundle = self.train_ray_generator(ray_indices)

        # Optionally provide unseen rays (from eval split when available)
        unseen_n = getattr(self.config, "unseen_num_rays_per_batch", 0)
        if unseen_n and unseen_n > 0:
            try:
                image_batch_eval = next(self.iter_eval_image_dataloader)
                sampler = self.eval_pixel_sampler
                generator = self.eval_ray_generator
            except StopIteration:
                # fallback to train if eval exhausted
                image_batch_eval = image_batch
                sampler = self.train_pixel_sampler
                generator = self.train_ray_generator
            if sampler is None:
                sampler = self.train_pixel_sampler
                generator = self.train_ray_generator
                image_batch_eval = image_batch
            eval_sample = sampler.sample(image_batch_eval)
            eval_sample = self._slice_batch(eval_sample, min(unseen_n, eval_sample["indices"].shape[0]))
            unseen_indices = eval_sample["indices"]
            unseen_ray_bundle = generator(unseen_indices)
            unseen_ray_bundle.nears = torch.ones_like(unseen_ray_bundle.origins[:, 0:1]) * 2
            unseen_ray_bundle.fars = torch.ones_like(unseen_ray_bundle.origins[:, 0:1]) * 6
            # attach unseen ray bundle; no GT needed
            batch["unseen_ray_bundle"] = unseen_ray_bundle

        return ray_bundle, batch
