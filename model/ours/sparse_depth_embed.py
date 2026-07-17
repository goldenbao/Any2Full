# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/main/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

from functools import partial
import math
import logging
from typing import Sequence, Tuple, Union, Callable
from typing import Callable, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch.utils.checkpoint
from torch.nn.init import trunc_normal_

from model.ours.depth_anything_v2.dinov2_layers import Mlp, PatchEmbed, SwiGLUFFNFused, MemEffAttention, NestedTensorBlock as Block

from model.ours.depth_anything_v2.dinov2 import DinoVisionTransformer


def make_2tuple(x):
    if isinstance(x, tuple):
        assert len(x) == 2
        return x

    assert isinstance(x, int)
    return (x, x)



class DepthPatchEmbed(nn.Module):
    """
    Sparse depth with Mask embedding: (B,1,H,W) -> (B,C,H,W)

    Args:
        embed_dim: Number of linear projection output channels.
        norm_layer: Normalization layer.
    """

    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Union[int, Tuple[int, int]] = 16,
        in_chans: int = 1,
        embed_dim: int = 768,
        norm_layer: Optional[Callable] = None,
        flatten_embedding: bool = True,
    ) -> None:
        super().__init__()

        image_HW = make_2tuple(img_size)
        patch_HW = make_2tuple(patch_size)
        patch_grid_size = (
            image_HW[0] // patch_HW[0],
            image_HW[1] // patch_HW[1],
        )

        self.img_size = image_HW
        self.patch_size = patch_HW
        self.patches_resolution = patch_grid_size
        self.num_patches = patch_grid_size[0] * patch_grid_size[1]
        self.embed_dim = embed_dim
        self.patch_size = patch_size
   
        
        # Conv2d(kernel_size=stride) → output 1x1, 等价于 Linear.
        # 用 Linear 避免 B*N 的 batch 展开，保持 batch=1，方便 NPU 部署。
        self.depth_heatmap_encoder = nn.ModuleList([
            nn.Linear(2 * self.patch_size * self.patch_size, embed_dim),
            nn.Linear(2 * (self.patch_size // 2) * (self.patch_size // 2), embed_dim),
            nn.Linear(2 * (self.patch_size // 4) * (self.patch_size // 4), embed_dim),
        ])

    
    
    def efficient_nearest_fill(self, depth_heatmap, mask_interp, epsilon=1e-5):
        now_dtype = depth_heatmap.dtype
        # Mask valid depth values
        valid_mask = (mask_interp >= epsilon)
        
        # Kernel for dilation
        kernel = torch.ones(3, 3).to(device=depth_heatmap.device, dtype=now_dtype)
        
        while not valid_mask.all():
            # Dilate valid regions
            expanded_mask = F.conv2d(valid_mask.to(now_dtype), kernel.unsqueeze(0).unsqueeze(0), 
                                    padding=1) > 0
            
            # Compute newly expanded regions
            new_valid = expanded_mask & (~valid_mask)
            
            if new_valid.any():
                # Fill newly expanded regions
                new_values = F.conv2d(depth_heatmap * valid_mask.to(now_dtype), kernel.unsqueeze(0).unsqueeze(0), 
                                    padding=1) / F.conv2d(valid_mask.to(now_dtype), kernel.unsqueeze(0).unsqueeze(0), 
                                                            padding=1)
                depth_heatmap = torch.where(new_valid, new_values, depth_heatmap)
                valid_mask = expanded_mask
            else:
                # Stop if no expansion
                break
        return depth_heatmap   
    
    def build_patch_depth_heatmap(self, x, patch_size, grid_size_list, epsilon=1e-6):
        """
        Args:
            x: (B, 1, H, W)  # normalized sparse depth, zeros are min values
        Returns:
            depth_heatmap: (B, N, 1, k, k)
            mask_heatmap:  (B, N, 1, k, k)
        """
        B, _, H, W = x.shape
        d_min = x.amin(dim=[1,2,3], keepdim=True)  # (B, 1, 1, 1)
        mask = (x > d_min + epsilon).to(x.dtype)

        h_patches = H // patch_size
        w_patches = W // patch_size
        N = h_patches * w_patches

        # Missing-aware interpolation (全图尺度处理，避免 B*N 展开)
        depth_vect_list=[]
        for i, gs in enumerate(grid_size_list):
            # 在全图上做 adaptive_avg_pool2d，等价于逐 patch pool 到 (gs, gs)
            depth_pooled = F.adaptive_avg_pool2d(x * mask, (h_patches * gs, w_patches * gs))
            mask_pooled = F.adaptive_avg_pool2d(mask, (h_patches * gs, w_patches * gs))
            depth_heatmap = depth_pooled / (mask_pooled + 1e-5)
            mask_interp = mask_pooled

            with torch.no_grad():
                depth_heatmap = self.efficient_nearest_fill(depth_heatmap, mask_interp)

            # (B, 1, h_p*gs, w_p*gs) → (B, N, 1, gs, gs)
            depth_heatmap = depth_heatmap.view(B, 1, h_patches, gs, w_patches, gs) \
                .permute(0, 2, 4, 1, 3, 5).reshape(B, N, 1, gs, gs)
            mask_heatmap = mask_interp.view(B, 1, h_patches, gs, w_patches, gs) \
                .permute(0, 2, 4, 1, 3, 5).reshape(B, N, 1, gs, gs)

            combined = torch.cat([depth_heatmap, mask_heatmap], dim=2)  # (B, N, 2, gs, gs)
            depth_vect = self.depth_heatmap_encoder[i](combined.view(B, N, -1))  # (B, N, embed_dim)
            depth_vect_list.append(depth_vect)
        return torch.sum(torch.stack(depth_vect_list, dim=0), dim=0), \
            (mask_heatmap.reshape(B, N, -1).sum(dim=-1, keepdim=True) > 0).to(mask_heatmap.dtype)


    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        # 兼容旧版 Conv2d checkpoint: 将 4D conv weight 展开为 2D linear weight
        for i in range(len(self.depth_heatmap_encoder)):
            w_key = prefix + f'depth_heatmap_encoder.{i}.weight'
            if w_key in state_dict and state_dict[w_key].dim() == 4:
                state_dict[w_key] = state_dict[w_key].view(self.embed_dim, -1)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        x_feat,patch_mask = self.build_patch_depth_heatmap(x, self.patch_size, grid_size_list=[self.patch_size,self.patch_size//2,self.patch_size//4])
        return x_feat, patch_mask #, patch_conf


class SparseDepthEmbed(nn.Module):
    def __init__(
        self,
        patch_size=14,
        init_chans=3,
        embed_dim=384,
        num_heads=4
    ):
        super().__init__()
  

        self.patch_size=patch_size
        self.patch_embed=DepthPatchEmbed(patch_size=patch_size, in_chans=1, embed_dim=embed_dim)
    
        self.num_tokens = 1
        
        self.interpolate_offset = 0.1
        self.interpolate_antialias=False
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        
        self.rgbd_proj_s= nn.Sequential(
            nn.Linear(embed_dim*2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.rgbd_proj_b= nn.Sequential(
            nn.Linear(embed_dim*2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
       
        
    def interpolate_pos_encoding(self, x, w, h):
        previous_dtype = x.dtype
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        pos_embed = self.pos_embed.float()
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        # Add a small value to avoid interpolation edge cases
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        # DINOv2 with register modifies interpolate_offset from 0.1 to 0.0
        w0, h0 = w0 + self.interpolate_offset, h0 + self.interpolate_offset
        # w0, h0 = w0 + 0.1, h0 + 0.1
        
        sqrt_N = math.sqrt(N)
        sx, sy = float(w0) / sqrt_N, float(h0) / sqrt_N
        
        if torch.__version__ >= '2.0.0':
            patch_pos_embed = nn.functional.interpolate(
                patch_pos_embed.reshape(1, int(sqrt_N), int(sqrt_N), dim).permute(0, 3, 1, 2),
                scale_factor=(sx, sy),
                mode="bicubic",
                antialias=self.interpolate_antialias
            )
        else:
            patch_pos_embed = nn.functional.interpolate(
                patch_pos_embed.reshape(1, int(sqrt_N), int(sqrt_N), dim).permute(0, 3, 1, 2),
                scale_factor=(sx, sy),
                mode="bicubic",
            )
        
        assert int(w0) == patch_pos_embed.shape[-2]
        assert int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(previous_dtype)
    
    def prepare_tokens_with_masks(self, x, rgb_feat_l, rgb_feat_h=None):
        B, _, W, H = x.shape
        x, mask = self.patch_embed(x)

        x_mean = x.mean(dim=1,keepdim=True)
        x = torch.cat((x_mean, x), dim=1)
        mask = torch.cat((torch.ones([1,1]).expand(x.shape[0], -1, -1).to(mask), mask), dim=1)
        
        x = x + self.interpolate_pos_encoding(x, W, H)
        

        rgb_feat=rgb_feat_l.detach()
        
        x = rgb_feat*(self.rgbd_proj_s(torch.cat((rgb_feat,x),dim=-1))+1)+self.rgbd_proj_b(torch.cat((rgb_feat,x),dim=-1))

        return x, mask
    

    def forward(self, x, rgb_feat_l, rgb_feat_h=None):
        x, mask=self.prepare_tokens_with_masks(x, rgb_feat_l, rgb_feat_h=None)
        
        return x, mask

   



