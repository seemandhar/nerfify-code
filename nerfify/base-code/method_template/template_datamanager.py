"""
Template DataManager

Extends VanillaDataManager with extension points for paper-specific
ray sampling strategies (unseen rays, pixel area, etc.).
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
class TemplateDataManagerConfig(VanillaDataManagerConfig):
    """DataManager configuration. Add paper-specific sampling parameters here."""

    _target: Type = field(default_factory=lambda: TemplateDataManager)


class TemplateDataManager(VanillaDataManager):
    """DataManager with extension points for paper-specific sampling.

    Override next_train() to add extra data to the batch dict
    (unseen rays, pixel area, etc.).
    """

    config: TemplateDataManagerConfig

    def __init__(
        self,
        config: TemplateDataManagerConfig,
        device: Union[torch.device, str] = "cpu",
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        **kwargs,
    ):
        super().__init__(
            config=config,
            device=device,
            test_mode=test_mode,
            world_size=world_size,
            local_rank=local_rank,
            **kwargs,
        )

    def next_train(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next training batch."""
        self.train_count += 1
        image_batch = next(self.iter_train_image_dataloader)
        assert self.train_pixel_sampler is not None
        assert isinstance(image_batch, dict)
        batch = self.train_pixel_sampler.sample(image_batch)
        ray_indices = batch["indices"]
        ray_bundle = self.train_ray_generator(ray_indices)
        return ray_bundle, batch

    def next_eval(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next eval batch."""
        self.eval_count += 1
        image_batch = next(self.iter_eval_image_dataloader)
        assert self.eval_pixel_sampler is not None
        assert isinstance(image_batch, dict)
        batch = self.eval_pixel_sampler.sample(image_batch)
        ray_indices = batch["indices"]
        ray_bundle = self.eval_ray_generator(ray_indices)
        return ray_bundle, batch
