"""
Nerfstudio Template Pipeline
"""

import typing
from dataclasses import dataclass, field
from typing import Literal, Optional, Type

import torch.distributed as dist
from torch.cuda.amp.grad_scaler import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP

from nerfify.methods.ers.enerf.enerf_datamanager import ENeRFDataManagerConfig
from nerfify.methods.ers.enerf.enerf_model import ENeRFModelConfig, ENeRFModel
from nerfstudio.data.datamanagers.base_datamanager import (
    DataManager,
    DataManagerConfig,
)
from nerfstudio.models.base_model import ModelConfig
from nerfstudio.pipelines.base_pipeline import (
    VanillaPipeline,
    VanillaPipelineConfig,
)
from nerfstudio.utils import profiler
import torch
from .losses.evaluator import Evaluator

@dataclass
class ENeRFPipelineConfig(VanillaPipelineConfig):
    """Configuration for pipeline instantiation"""

    _target: Type = field(default_factory=lambda: ENeRFPipeline)
    """target class to instantiate"""
    datamanager: DataManagerConfig = field(default_factory=ENeRFDataManagerConfig)
    """specifies the datamanager config"""
    model: ModelConfig = field(default_factory=ENeRFModelConfig)
    """specifies the model config"""


class ENeRFPipeline(VanillaPipeline):
    """ENeRF Pipeline

    Args:
        config: the pipeline config used to instantiate class
    """

    def __init__(
        self,
        config: ENeRFPipelineConfig,
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
            device=device, test_mode=test_mode, world_size=world_size, local_rank=local_rank
        )
        # self.datamanager.to(device)

        assert self.datamanager.train_dataset is not None, "Missing input dataset"
        self._model = config.model.setup(
            # scene_box=self.datamanager.train_dataset.scene_box,
            num_train_data=len(self.datamanager.train_dataset),
            # metadata=self.datamanager.train_dataset.metadata,
            device=device,
            grad_scaler=grad_scaler,
        )
        self.model.to(device)

        self.world_size = world_size
        if world_size > 1:
            self._model = typing.cast(
                ENeRFModel, DDP(self._model, device_ids=[local_rank], find_unused_parameters=True)
            )
            dist.barrier(device_ids=[local_rank])

        self.evaluator = Evaluator()
            
    def to_cuda(self, batch, device=torch.device('cuda:0')):
        if isinstance(batch, tuple) or isinstance(batch, list):
            #batch[k] = [b.cuda() for b in batch[k]]
            #batch[k] = [b.to(self.device) for b in batch[k]]
            batch = [self.to_cuda(b, device) for b in batch]
        elif isinstance(batch, dict):
            #batch[k] = {key: self.to_cuda(batch[k][key]) for key in batch[k]}
            batch_ = {}
            for key in batch:
                if key == 'meta':
                    batch_[key] = batch[key]
                else:
                    batch_[key] = self.to_cuda(batch[key], device)
            batch = batch_
        else:
            # batch[k] = batch[k].cuda()
            batch = batch.to(device)
        return batch

    @profiler.time_function
    def get_train_loss_dict(self, step: int):
        batch = self.datamanager.next_train(step)
        batch = self.to_cuda(batch, self.device)
        batch['step'] = 0
        model_outputs = self.model(batch)
        loss, scalar_stats = self.model.compute_loss_and_metrics(model_outputs, batch)
        loss_dict = {'loss': loss}
        # metrics_dict = self.model.get_metrics_dict(model_outputs, batch)
        # loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)
        return model_outputs, loss_dict, scalar_stats # we return model_outputs, loss_dict, metrics_dict (scalar_stats)
    
    @profiler.time_function
    def get_eval_loss_dict(self, step: int):
        self.eval()
        idx = 0
        while(idx < len(self.datamanager.eval_dataset)):
            batch = self.datamanager.next_eval(step) # batch size is 1
            # batch = self.to_cuda(batch, self.device)
            for k in batch:
                if k != 'meta':
                    batch[k] = batch[k].cuda()
            # batch['step'] = 0
            with torch.no_grad():
                model_outputs = self.model(batch)
            
            self.evaluator.evaluate(model_outputs, batch)
            idx += 1
        self.evaluator.summarize()
        # loss, scalar_stats = self.model.compute_loss_and_metrics(model_outputs, batch)
        # loss_dict = {'loss': loss}
        self.train()
        # metrics_dict = self.model.get_metrics_dict(model_outputs, batch)
        # loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)
        return model_outputs, {}, {} # we return model_outputs, loss_dict, metrics_dict (scalar_stats)

    def get_eval_image_metrics_and_images(self, step):
        self.eval()
        idx = 0
        while(idx < len(self.datamanager.eval_dataset)):
            batch = self.datamanager.next_eval(step) # batch size is 1
            # batch = self.to_cuda(batch, self.device)
            for k in batch:
                if k != 'meta':
                    batch[k] = batch[k].cuda()
            # batch['step'] = 0
            with torch.no_grad():
                model_outputs = self.model(batch)
            
            self.evaluator.evaluate(model_outputs, batch, output_path=None)
            idx += 1
        
        ret = self.evaluator.summarize()
        self.train()

        return ret, {}

    def get_average_eval_image_metrics(self, step, output_path):
        self.eval()
        idx = 0
        while(idx < len(self.datamanager.eval_dataset)):
            batch = self.datamanager.next_eval(step) # batch size is 1
            # batch = self.to_cuda(batch, self.device)
            for k in batch:
                if k != 'meta':
                    batch[k] = batch[k].cuda()
            # batch['step'] = 0
            with torch.no_grad():
                model_outputs = self.model(batch)
            
            self.evaluator.evaluate(model_outputs, batch, output_path)
            idx += 1
        
        ret = self.evaluator.summarize()
        self.train()

        return ret

    def get_param_groups(self):
        """Get the param groups for the pipeline.

        Returns:
            A list of dictionaries containing the pipeline's param groups.
        """
        model_params = self.model.get_param_groups()
        return {**model_params}
