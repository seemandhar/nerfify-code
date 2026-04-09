"""
Template DataManager
"""

from dataclasses import dataclass, field
from typing import Dict, Literal, Tuple, Type, Union

import torch

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.datamanagers.base_datamanager import (
    VanillaDataManager,
    VanillaDataManagerConfig,
)
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes, TrainingCallbackLocation



@dataclass
class AnisNeRFDataManagerConfig(VanillaDataManagerConfig):
    """Template DataManager Config

    Add your custom datamanager config parameters here.
    """

    _target: Type = field(default_factory=lambda: AnisNeRFDataManager)
    sample_annealing = False
    start_samples = 1024
    end_samples = 4096
    max_steps = 1000


class AnisNeRFDataManager(VanillaDataManager):
    """BioNeRF DataManager

    Args:
        config: the DataManagerConfig used to instantiate class
    """

    config: AnisNeRFDataManagerConfig

    def __init__(
        self,
        config: AnisNeRFDataManagerConfig,
        device: Union[torch.device, str] = "cpu",
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        **kwargs,  # pylint: disable=unused-argument
    ):
        super().__init__(
            config=config, device=device, test_mode=test_mode, world_size=world_size, local_rank=local_rank, **kwargs
        )
        self.start_samples = config.start_samples
        self.end_samples = config.end_samples
        self.max_steps = config.max_steps

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ):
        callbacks = super().get_training_callbacks(training_callback_attributes)

        if self.config.sample_annealing:
            def step(step):
                # print(f"{self.train_pixel_sampler.num_rays_per_batch}")
                progress = min(step / self.max_steps, 1.0)
                new_samples = int(
                    self.start_samples + progress * (self.end_samples - self.start_samples)
                )
                self.train_pixel_sampler.set_num_rays_per_batch(new_samples)

            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=step,
                )
            )

        return callbacks

    def next_train(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the train dataloader."""
        self.train_count += 1
        image_batch = next(self.iter_train_image_dataloader)
        assert self.train_pixel_sampler is not None
        assert isinstance(image_batch, dict)
        batch = self.train_pixel_sampler.sample(image_batch)
        ray_indices = batch["indices"]
        ray_bundle = self.train_ray_generator(ray_indices)
        return ray_bundle, batch