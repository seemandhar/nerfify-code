"""
Template DataManager
"""

from dataclasses import dataclass, field
from typing import Dict, Literal, Tuple, Type, Union, List

import torch

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.datamanagers.base_datamanager import (
    VanillaDataManager,
    VanillaDataManagerConfig,
)

from torch.nn import Parameter


import numpy as np
import os
from . import enerf_utils
from .config import cfg
from itertools import cycle
import imageio
import tqdm
from multiprocessing import Pool
import copy
import cv2
import random
from .config import cfg
from .all_utils import data_utils
import torch
from .sampler_utils.make_dataset import make_data_loader

if cfg.fix_random:
    random.seed(0)
    np.random.seed(0)

def make_dataset(cfg, is_train=True):
    if is_train:
        args = cfg.train_dataset
        module = cfg.train_dataset_module
        path = cfg.train_dataset_path
    else:
        args = cfg.test_dataset
        module = cfg.test_dataset_module
        path = cfg.test_dataset_path
    dataset = Dataset(**args)
    return dataset


class Dataset:
    def __init__(self, **kwargs):
        super(Dataset, self).__init__()
        self.data_root = os.path.join(".", kwargs['data_root'])
        self.split = kwargs['split']
        if 'scene' in kwargs:
            self.scenes = [kwargs['scene']]
        else:
            self.scenes = []
        self.build_metas(kwargs['ann_file'])
        self.depth_ranges = [425., 905.]

    def build_metas(self, ann_file):
        scenes = [line.strip() for line in open(ann_file).readlines()]
        dtu_pairs = torch.load('/home/keshav06/ENeRF/data/mvsnerf/pairs.th')

        self.scene_infos = {}
        self.metas = []
        if len(self.scenes) != 0:
            scenes = self.scenes

        for scene in scenes:
            scene_info = {'ixts': [], 'exts': [], 'dpt_paths': [], 'img_paths': []}
            for i in range(49):
                cam_path = os.path.join(self.data_root, 'Cameras/train/{:08d}_cam.txt'.format(i))
                ixt, ext, _ = data_utils.read_cam_file(cam_path)
                ext[:3, 3] = ext[:3, 3]
                ixt[:2] = ixt[:2] * 4
                dpt_path = os.path.join(self.data_root, 'Depths/{}_train/depth_map_{:04d}.pfm'.format(scene, i))
                img_path = os.path.join(self.data_root, 'Rectified/{}_train/rect_{:03d}_3_r5000.png'.format(scene, i+1))
                scene_info['ixts'].append(ixt.astype(np.float32))
                scene_info['exts'].append(ext.astype(np.float32))
                scene_info['dpt_paths'].append(dpt_path)
                scene_info['img_paths'].append(img_path)

            if self.split == 'train' and len(self.scenes) != 1:
                train_ids = np.arange(49).tolist()
                test_ids = np.arange(49).tolist()
            elif self.split == 'train' and len(self.scenes) == 1:
                train_ids = dtu_pairs['dtu_train']
                test_ids = dtu_pairs['dtu_train']
            else:
                train_ids = dtu_pairs['dtu_train']
                test_ids = dtu_pairs['dtu_val']
            scene_info.update({'train_ids': train_ids, 'test_ids': test_ids})
            self.scene_infos[scene] = scene_info

            cam_points = np.array([np.linalg.inv(scene_info['exts'][i])[:3, 3] for i in train_ids])
            for tar_view in test_ids:
                cam_point = np.linalg.inv(scene_info['exts'][tar_view])[:3, 3]
                distance = np.linalg.norm(cam_points - cam_point[None], axis=-1)
                argsorts = distance.argsort()
                argsorts = argsorts[1:] if tar_view in train_ids else argsorts
                input_views_num = cfg.enerf.train_input_views[1] + 1 if self.split == 'train' else cfg.enerf.test_input_views
                src_views = [train_ids[i] for i in argsorts[:input_views_num]]
                self.metas += [(scene, tar_view, src_views)]

    def __getitem__(self, index_meta):
        index, input_views_num = index_meta
        scene, tar_view, src_views = self.metas[index]
        if self.split == 'train':
            if random.random() < 0.1:
                src_views = src_views + [tar_view]
            src_views = random.sample(src_views[:input_views_num+1], input_views_num)
        scene_info = self.scene_infos[scene]

        tar_img = np.array(imageio.imread(scene_info['img_paths'][tar_view])) / 255.
        H, W = tar_img.shape[:2]
        tar_ext, tar_ixt = scene_info['exts'][tar_view], scene_info['ixts'][tar_view]
        if self.split != 'train': # only used for evaluation
            tar_dpt = data_utils.read_pfm(scene_info['dpt_paths'][tar_view])[0].astype(np.float32)
            # tar_dpt = cv2.resize(tar_dpt, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
            # tar_dpt = tar_dpt[44:556, 80:720]
            tar_dpt = cv2.resize(tar_dpt, (640, 512), interpolation=cv2.INTER_NEAREST)
            tar_mask = (tar_dpt > 0.).astype(np.uint8)
        else:
            tar_dpt = np.ones_like(tar_img)
            tar_mask = np.ones_like(tar_img)

        src_inps, src_exts, src_ixts = self.read_src(scene_info, src_views)

        ret = {'src_inps': src_inps,
               'src_exts': src_exts,
               'src_ixts': src_ixts}
        ret.update({'tar_ext': tar_ext,
                    'tar_ixt': tar_ixt})
        if self.split != 'train':
            ret.update({'tar_img': tar_img,
                        'tar_dpt': tar_dpt,
                        'tar_mask': tar_mask})
        else: # CHANGED
            ret.update({'tar_img': tar_img,
                        'tar_dpt': tar_dpt,
                        'tar_mask': tar_mask})
            
        ret.update({'near_far': np.array(self.depth_ranges).astype(np.float32)})
        ret.update({'meta': {'scene': scene, 'tar_view': tar_view, 'frame_id': 0, 'split' : self.split}})
        
        if(self.split != 'train'):
            # for the eval images, sample normally (i.e the entire image)
            for i in range(cfg.enerf.cas_config.num):
                rays, rgb, msk = enerf_utils.build_rays(tar_img, tar_ext, tar_ixt, tar_mask, i, self.split)
                s = cfg.enerf.cas_config.volume_scale[i]
                if self.split != 'train': # evaluation
                    tar_dpt_i = cv2.resize(tar_dpt, None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST)
                    ret.update({f'tar_dpt_{i}': tar_dpt_i.astype(np.float32)})
                ret.update({f'rays_{i}': rays, f'rgb_{i}': rgb.astype(np.float32), f'msk_{i}': msk})
                ret['meta'].update({f'h_{i}': H, f'w_{i}': W})
        else:
            for i in range(cfg.enerf.cas_config.num):
                # for the train images, we will sample the rays inside the network itself, because that depends on the depth as well
                scale = cfg.enerf.cas_config.render_scale[i]
                # if scale != 1.:
                tar_img_i = cv2.resize(tar_img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                tar_img_i_grey = cv2.cvtColor(tar_img_i.astype(np.float32), cv2.COLOR_BGR2GRAY)
                tar_msk_i = cv2.resize(tar_mask, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
                tar_ixt_i = tar_ixt.copy()
                tar_ixt_i[:2] *= scale

                # rays, rgb, msk = enerf_utils.build_rays(tar_img, tar_ext, tar_ixt, tar_mask, i, self.split)

                s = cfg.enerf.cas_config.volume_scale[i]
                # this will be done in the network itself
                # ret.update({f'rays_{i}': rays, f'rgb_{i}': rgb.astype(np.float32), f'msk_{i}': msk}) 
                ret.update({f'tar_img_{i}': tar_img_i.astype(np.float32), f'tar_img_{i}_grey': tar_img_i_grey.astype(np.float32), f'tar_msk_{i}': tar_msk_i, f'tar_ixt_{i}': tar_ixt_i})
                ret['meta'].update({f'h_{i}': H, f'w_{i}': W})
        return ret

    def read_src(self, scene_info, src_views):
        inps, exts, ixts = [], [], []
        for src_view in src_views:
            inps.append((np.array(imageio.imread(scene_info['img_paths'][src_view])) / 255.) * 2. - 1.)
            exts.append(scene_info['exts'][src_view])
            ixts.append(scene_info['ixts'][src_view])
        return np.stack(inps).transpose((0, 3, 1, 2)).astype(np.float32), np.stack(exts), np.stack(ixts)

    def __len__(self):
        return len(self.metas)


@dataclass
class ENeRFDataManagerConfig(VanillaDataManagerConfig):
    """Template DataManager Config

    Add your custom datamanager config parameters here.
    """

    _target: Type = field(default_factory=lambda: ENerfDataManager)


class ENerfDataManager():
    """ENerf DataManager

    Args:
        config: the DataManagerConfig used to instantiate class
    """

    config: ENeRFDataManagerConfig

    def __init__(
        self,
        config: ENeRFDataManagerConfig,
        device: Union[torch.device, str] = "cpu",
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        **kwargs,  # pylint: disable=unused-argument
    ):
        # super().__init__(
        #     config=config, device=device, test_mode=test_mode, world_size=world_size, local_rank=local_rank, **kwargs
        # )
        self.config = config
        self.train_count = 0
        self.eval_count = 0

        self.train_dataset = make_dataset(cfg, is_train=True)
        self.train_data_loader = make_data_loader(cfg, self.train_dataset, is_train=True, is_distributed=(world_size>1))
        self.train_iter = cycle(self.train_data_loader)

        self.eval_dataset = make_dataset(cfg, is_train=False)
        self.eval_data_loader = make_data_loader(cfg, self.eval_dataset, is_train=False)
        self.eval_iter = cycle(self.eval_data_loader)

    def next_train(self, step):
        """Returns the next batch of data from the train dataloader."""
        self.train_count += 1
        # batch = next(self.train_dataset)
        batch = next(self.train_iter)
        # batch = self.train_pixel_sampler.sample(image_batch)
        # ray_indices = batch["indices"]
        # ray_bundle = self.train_ray_generator(ray_indices)
        return batch
    
    def next_eval(self, step):
        """Returns the next batch of data from the train dataloader."""
        self.eval_count += 1
        # batch = next(self.train_dataset)
        batch = next(self.eval_iter)
        # batch = self.train_pixel_sampler.sample(image_batch)
        # ray_indices = batch["indices"]
        # ray_bundle = self.train_ray_generator(ray_indices)
        return batch
    
    def get_training_callbacks(self, attr_name):
        """Get any training callbacks the datamanager uses.

        Returns:
            A list of training callbacks.
        """
        return []

    def get_train_rays_per_batch(self):
        return 1