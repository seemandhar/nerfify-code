"""
Template Pipeline — proper DDP init and clean model/datamanager wiring.

Most papers do NOT need to modify the pipeline. Customization happens
in the datamanager and model components.
"""
import typing
from dataclasses import dataclass, field
from typing import Literal, Optional, Type

import torch.distributed as dist
from torch.cuda.amp.grad_scaler import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP

from .template_datamanager import TemplateDataManagerConfig
from .template_model import TemplateModel, TemplateModelConfig
from nerfstudio.data.datamanagers.base_datamanager import DataManager, DataManagerConfig
from nerfstudio.models.base_model import ModelConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipeline, VanillaPipelineConfig


@dataclass
class TemplatePipelineConfig(VanillaPipelineConfig):
    """Pipeline configuration."""

    _target: Type = field(default_factory=lambda: TemplatePipeline)
    datamanager: DataManagerConfig = TemplateDataManagerConfig()
    model: ModelConfig = TemplateModelConfig()


class TemplatePipeline(VanillaPipeline):
    """Pipeline with explicit construction and proper DDP init."""

    def __init__(
        self,
        config: TemplatePipelineConfig,
        device: str,
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        grad_scaler: Optional[GradScaler] = None,
    ):
        super(VanillaPipeline, self).__init__()
        self.config = config
        self.test_mode = test_mode

        self.datamanager: DataManager = config.datamanager.setup(
            device=device,
            test_mode=test_mode,
            world_size=world_size,
            local_rank=local_rank,
        )

        assert self.datamanager.train_dataset is not None, "Missing input dataset"
        self._model = config.model.setup(
            scene_box=self.datamanager.train_dataset.scene_box,
            num_train_data=len(self.datamanager.train_dataset),
            metadata=self.datamanager.train_dataset.metadata,
            device=device,
            grad_scaler=grad_scaler,
        )
        self.model.to(device)

        self.world_size = world_size
        if world_size > 1:
            self._model = typing.cast(
                TemplateModel,
                DDP(self._model, device_ids=[local_rank], find_unused_parameters=True),
            )
            if dist.is_initialized():
                dist.barrier()
