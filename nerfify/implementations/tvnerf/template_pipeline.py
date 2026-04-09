"""
TVNeRF Pipeline — clean wiring for model/datamanager with proper DDP init.
"""
import typing
from dataclasses import dataclass, field
from typing import Literal, Optional, Type

import torch.distributed as dist
from torch.cuda.amp.grad_scaler import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP

from nerfify.methods.tvnerf.template_datamanager import TvnerfDataManagerConfig
from nerfify.methods.tvnerf.template_model import TvnerfModel, TvnerfModelConfig
from nerfstudio.data.datamanagers.base_datamanager import DataManager, DataManagerConfig
from nerfstudio.models.base_model import ModelConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipeline, VanillaPipelineConfig


@dataclass
class TvnerfPipelineConfig(VanillaPipelineConfig):
    """Configuration for pipeline instantiation"""
    _target: Type = field(default_factory=lambda: TvnerfPipeline)
    datamanager: DataManagerConfig = TvnerfDataManagerConfig()
    model: ModelConfig = TvnerfModelConfig()


class TvnerfPipeline(VanillaPipeline):
    """Vanilla pipeline with explicit construction and proper DDP init."""

    def __init__(
        self,
        config: TvnerfPipelineConfig,
        device: str,
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        grad_scaler: Optional[GradScaler] = None,
    ):
        super(VanillaPipeline, self).__init__()
        self.config = config
        self.test_mode = test_mode

        # Data
        self.datamanager: DataManager = config.datamanager.setup(
            device=device, test_mode=test_mode, world_size=world_size, local_rank=local_rank
        )
        # self.datamanager.to(device)

        # Model
        assert self.datamanager.train_dataset is not None, "Missing input dataset"
        self._model = config.model.setup(
            scene_box=self.datamanager.train_dataset.scene_box,
            num_train_data=len(self.datamanager.train_dataset),
            metadata=self.datamanager.train_dataset.metadata,
            device=device,
            grad_scaler=grad_scaler,
        )
        self.model.to(device)

        # DDP (if multi-GPU)
        self.world_size = world_size
        if world_size > 1:
            self._model = typing.cast(
                TvnerfModel, DDP(self._model, device_ids=[local_rank], find_unused_parameters=True)
            )
            if dist.is_initialized():
                dist.barrier()
