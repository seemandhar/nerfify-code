import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import cv2
from .feature_net import FeatureNet
from .cost_reg_net import CostRegNet, MinCostRegNet
from . import utils
from .config import cfg
from .nerf import NeRF

class Network(nn.Module):
    def __init__(self,):
        super(Network, self).__init__()
        self.feature_net = FeatureNet()
        for i in range(cfg.enerf.cas_config.num):
            if i == 0:
                cost_reg_l = MinCostRegNet(int(32 * (2**(-i))))
            else:
                cost_reg_l = CostRegNet(int(32 * (2**(-i))))
            setattr(self, f'cost_reg_{i}', cost_reg_l)
            nerf_l = NeRF(feat_ch=cfg.enerf.cas_config.nerf_model_feat_ch[i]+3)
            setattr(self, f'nerf_{i}', nerf_l)

    def render_rays(self, rays, **kwargs):
        level, batch, im_feat, feat_volume, nerf_model = kwargs['level'], kwargs['batch'], kwargs['im_feat'], kwargs['feature_volume'], kwargs['nerf_model']
        world_xyz, uvd, z_vals = utils.sample_along_depth(rays, N_samples=cfg.enerf.cas_config.num_samples[level], level=level)
        B, N_rays, N_samples = world_xyz.shape[:3]
        rgbs = utils.unpreprocess(batch['src_inps'], render_scale=cfg.enerf.cas_config.render_scale[level])
        up_feat_scale = cfg.enerf.cas_config.render_scale[level] / cfg.enerf.cas_config.im_ibr_scale[level]
        if up_feat_scale != 1.:
            B, S, C, H, W = im_feat.shape
            im_feat = F.interpolate(im_feat.reshape(B*S, C, H, W), None, scale_factor=up_feat_scale, align_corners=True, mode='bilinear').view(B, S, C, int(H*up_feat_scale), int(W*up_feat_scale))

        img_feat_rgb = torch.cat((im_feat, rgbs), dim=2)
        H_O, W_O = kwargs['batch']['src_inps'].shape[-2:]
        B, H, W = len(uvd), int(H_O * cfg.enerf.cas_config.render_scale[level]), int(W_O * cfg.enerf.cas_config.render_scale[level])
        uvd[..., 0], uvd[..., 1] = (uvd[..., 0]) / (W-1), (uvd[..., 1]) / (H-1)
        vox_feat = utils.get_vox_feat(uvd.reshape(B, -1, 3), feat_volume)
        img_feat_rgb_dir = utils.get_img_feat(world_xyz, img_feat_rgb, batch, self.training, level) # B * N * S * (8+3+4)
        net_output = nerf_model(vox_feat, img_feat_rgb_dir)
        net_output = net_output.reshape(B, -1, N_samples, net_output.shape[-1])
        outputs = utils.raw2outputs(net_output, z_vals, cfg.enerf.white_bkgd)
        return outputs

    def batchify_rays(self, rays, **kwargs):
        all_ret = {}
        chunk = cfg.enerf.chunk_size
        for i in range(0, rays.shape[1], chunk):
            ret = self.render_rays(rays[:, i:i + chunk], **kwargs)
            for k in ret:
                if k not in all_ret:
                    all_ret[k] = []
                all_ret[k].append(ret[k])
        all_ret = {k: torch.cat(all_ret[k], dim=1) for k in all_ret}
        return all_ret


    def forward_feat(self, x):
        B, S, C, H, W = x.shape
        x = x.view(B*S, C, H, W)
        feat2, feat1, feat0 = self.feature_net(x)
        feats = {
                'level_2': feat0.reshape((B, S, feat0.shape[1], H, W)),
                'level_1': feat1.reshape((B, S, feat1.shape[1], H//2, W//2)),
                'level_0': feat2.reshape((B, S, feat2.shape[1], H//4, W//4)),
                }
        return feats

    def forward_render(self, ret, batch):
        B, _, _, H, W = batch['src_inps'].shape
        rgb = ret['rgb'].reshape(B, H, W, 3).permute(0, 3, 1, 2)
        rgb = self.cnn_renderer(rgb)
        ret['rgb'] = rgb.permute(0, 2, 3, 1).reshape(B, H*W, 3)

    def calculate_variance_prob(self, x, n=3):
        # x is of shape (b, h, w). Our aim is to find the variance of each patch of size (n * n) (n = 3 in the paper). 
        mean_filter = torch.ones(n, n).to(x) / (n ** 2)
        mean_filter = mean_filter[None, None] # (1, 1, n, n)
        padding = (n-1) // 2
        x_avg2 = (F.conv2d(x, mean_filter, padding=padding)) ** 2
        x2_avg = F.conv2d(x**2, mean_filter, padding=padding)
        diff = torch.clamp(x2_avg - x_avg2, min=0)
        std = torch.sqrt(diff)
        min_val = torch.mean(std) * 0.01
        std = torch.clamp(std, min=min_val)
        std = std / torch.max(std)
        return std
        
    # NOTE : call this ONLY during training
    def sample_ers(self, batch, depth, level, beta=0.5):
        # here we are going to sample rays from the tar_img based on 1) depth and 2) colors of the image
        # we are going to return the rays, rgb and mask as the function in enerf_utils.py
        scale = cfg.enerf.cas_config.render_scale[level]
        # if scale != 1.:
        #     tar_img = cv2.resize(tar_img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        #     tar_msk = cv2.resize(tar_msk, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        #     tar_ixt = tar_ixt.copy()
        #     tar_ixt[:2] *= scale
        
        tar_img_grey = batch[f'tar_img_{level}_grey']
        tar_img = batch[f'tar_img_{level}']
        tar_msk = batch[f'tar_msk_{level}']
        tar_ixt = batch[f'tar_ixt_{level}']

        depth_upscaled = F.interpolate(depth[:, None], scale_factor=2, mode='bilinear', align_corners=True)[:, 0]

        B, H, W = tar_img_grey.shape # tar img is grayscale
        c2w = torch.linalg.inv(batch['tar_ext'])
        num_rays = cfg.enerf.cas_config.num_rays[level]

        color_prob = self.calculate_variance_prob(tar_img_grey, n=3) # (b, h, w)
        depth_prob = self.calculate_variance_prob(depth_upscaled, n=3) # (b, h, w)
        prob = color_prob * beta + depth_prob * (1 - beta)

        prob_flat = prob.reshape(tar_img.shape[0], -1)
        ys, xs = torch.meshgrid(torch.arange(H).to(prob), torch.arange(W).to(prob), indexing='ij')
        coords = torch.stack([xs, ys], dim=-1).view(-1, 2)  # (h*w, 2)

        sampled_coords = []
        for i in range(tar_img.shape[0]):
            idx = torch.multinomial(prob_flat[i], num_rays, replacement=True)
            sampled_coords.append(coords[idx])
        
        sampled_coords = torch.stack(sampled_coords, dim=0)  # (b, num_rays, 2)
        X = sampled_coords[:, :, 0]
        Y = sampled_coords[:, :, 1]

        rays_o = c2w[:, :3, 3][:, None].repeat(1, num_rays, 1)
        XYZ = torch.concatenate((X[..., None], Y[..., None], torch.ones_like(X[..., None])), axis=-1)
        XYZ = XYZ @ (torch.linalg.inv(tar_ixt).permute(0, 2, 1) @ c2w[:, :3, :3].permute(0, 2, 1))
        rays = torch.concatenate((rays_o, XYZ, X[..., None], Y[..., None]), axis=-1)
        Y = Y.to(torch.long)
        X = X.to(torch.long)

        assert len(Y[Y>=H]) == 0, "error"
        assert len(X[X>=W]) == 0, "error"

        batch_idx = torch.arange(B)[:, None].to(X)
        rgb = tar_img[batch_idx, Y, X]
        msk = tar_msk[batch_idx, Y, X]

        rays = rays.to(torch.float32).reshape(B, -1, 8)
        rgb = rgb.reshape(B, -1, 3)
        msk = msk.reshape(B, -1)
        
        batch.update({f'rays_{level}': rays, f'rgb_{level}': rgb.to(torch.float32), f'msk_{level}': msk})

        return batch

    def forward(self, batch):
        # print(batch.keys())
        feats = self.forward_feat(batch['src_inps'])
        ret = {}
        split = batch['meta']['split'][0]

        depth, std, near_far = None, None, None
        for i in range(cfg.enerf.cas_config.num):
            feature_volume, depth_values, near_far = utils.build_feature_volume(
                    feats[f'level_{i}'],
                    batch,
                    D=cfg.enerf.cas_config.volume_planes[i],
                    depth=depth,
                    std=std,
                    near_far=near_far,
                    level=i)

            # print(feature_volume.shape, depth_values.shape)
            feature_volume, depth_prob = getattr(self, f'cost_reg_{i}')(feature_volume)
            # print(feature_volume.shape, depth_values.shape)
            depth, std = utils.depth_regression(depth_prob, depth_values, i, batch) # depth is of shape (b, h//s, w//s)
            
            if(split == "train"):
                batch = self.sample_ers(batch, depth, i)
                # print(batch.keys())

            # print(depth.shape, std.shape)
            if not cfg.enerf.cas_config.render_if[i]:
                continue
            rays = utils.build_rays(depth, std, batch, self.training, near_far, i)
            # print(rays.shape)
            # UV(2) +  ray_o (3) + ray_d (3) + ray_near_far (2) + volume_near_far (2)
            im_feat_level = cfg.enerf.cas_config.render_im_feat_level[i]
            # print(im_feat_level)
            ret_i = self.batchify_rays(
                    rays=rays,
                    feature_volume=feature_volume,
                    batch=batch,
                    im_feat=feats[f'level_{im_feat_level}'],
                    nerf_model=getattr(self, f'nerf_{i}'),
                    level=i)

            # batchify rays function renders the rays in batches to avoid OOM, so ret_i['rgb'] is (B, N_rays, 3)
            # if i == 1:
                # self.forward_render(ret_i, batch)
            if cfg.enerf.cas_config.depth_inv[i]:
                ret_i.update({'depth_mvs': 1./depth})
            else:
                ret_i.update({'depth_mvs': depth})
            ret_i.update({'std': std})
            if ret_i['rgb'].isnan().any():
                __import__('ipdb').set_trace()
            ret.update({key+f'_level{i}': ret_i[key] for key in ret_i})
        return ret
