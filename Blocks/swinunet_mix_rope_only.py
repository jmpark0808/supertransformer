# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math
from Blocks.swin_common import PatchEmbed, BasicLayerUpsampleMA, BasicLayer, PatchMerging
from Blocks.swin_encoder_rope import SwinTransformer
from Blocks.swin_rope import BasicLayerRoPE
from Blocks.performer2 import Transformer
WindowProcess = None
WindowProcessReverse = None
print("[Warning] Fused window process have not been installed. Please refer to get_started.md for installation.")


    
class SwinUTransformer(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=[96, 96*2, 96*4, 96*8], depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, fused_window_process=False, **kwargs):
        super().__init__()


        swinencoder = SwinTransformer(img_size=img_size, patch_size=patch_size, in_chans=in_chans, num_classes=num_classes,
                                      embed_dim=embed_dim, depths=depths, num_heads=num_heads,
                                      window_size=window_size, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                      drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate,
                                      norm_layer=norm_layer, ape=ape, patch_norm=patch_norm, 
                                      use_checkpoint=use_checkpoint, fused_window_process=fused_window_process, **kwargs)
        self.img_size = img_size
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim[0]
        self.ape = ape
        self.patch_norm = patch_norm
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = swinencoder.patch_embed
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        # absolute position embedding


        self.pos_drop = nn.Dropout(p=drop_rate)

        
        # build layers
        self.layers = swinencoder.layers
        # resolutions.append(patches_resolution[0] // (2 ** i_layer))
        embed_dims = swinencoder.embed_dims
        resolutions = swinencoder.resolutions
        embed_dims.reverse()
            
            

        self.upsample_layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayerUpsampleMA(dim=embed_dims[i_layer],
                                       total_dim=sum(embed_dim),
                               input_resolution=resolutions,
                               num_heads=num_heads[self.num_layers-i_layer-1],
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, 
                               qk_scale=qk_scale, 
                               drop=drop_rate, 
                               attn_drop=attn_drop_rate,
                               use_checkpoint=use_checkpoint)
            self.upsample_layers.append(layer)

        self.upsample = nn.Upsample(size=img_size[0])
        self.sod_head = nn.Linear(sum(embed_dim), 8)
        self.intermediate_head = nn.Linear(sum(embed_dim), 1)
        self.shallow = nn.Conv2d(8, 1, 1)
        # self.locations = swinencoder.locations

        # self.refinement = BasicLayerRoPE(dim=4,
        #                           out_dim=4,
        #                        input_resolution=(224,
        #                                          224),
        #                        depth=4,
        #                        num_heads=1,
        #                        window_size=7,
        #                        mlp_ratio=2,
        #                        qkv_bias=qkv_bias, qk_scale=qk_scale,
        #                        drop=drop_rate, attn_drop=attn_drop_rate,
        #                        drop_path=0,
        #                        norm_layer=norm_layer,
        #                        downsample=None,
        #                        use_checkpoint=use_checkpoint,
        #                        fused_window_process=fused_window_process)
        # self.refinement = nn.Sequential(*[nn.Conv2d(1, 32*28*28, 28, 28),
        #                                      nn.ReLU(),
        #                                        nn.Conv2d(32*28*28, 28*28, 1)])
        self.refinement = Transformer(8, 2, 1, 1, 8, 16, 0, 0, 224)
        
        self.apply(self._init_weights)
       
        self.resolutions = resolutions.copy()
        self.resolutions.reverse()
        del swinencoder

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x):
        x = x[:, 2:, :, :]
        # centroids = x[:, :2, :, :].permute(0, 2, 3, 1)
        # centroids = self.locations(centroids)
        # centroids = centroids.reshape(centroids.size(0), -1, centroids.size(3))
        
        x = self.patch_embed(x)
        # x = x + centroids
        x = self.pos_drop(x)

     
        ft = []
        for layer in self.layers:
            ds, x = layer(x)
            ft.append(ds)


        up_ft = []
        for idx, layer in enumerate(self.upsample_layers):
            x = layer(ft[len(ft)-idx-1], ft)
            ft[len(ft)-idx-1] = x
            
            res = self.resolutions[idx]
            x = x.reshape(x.size(0), res[0], res[1], -1).permute(0, 3, 1, 2)
            x = self.upsample(x).permute(0, 2, 3, 1)
            x = x.reshape(x.size(0), self.img_size**2, -1)
            up_ft.append(x)

        up_ft = torch.cat(up_ft, dim=2)
        return up_ft



    def forward(self, x, segments):
        x = self.forward_features(x)
        intermediate = self.intermediate_head(x)

        x = self.sod_head(x)
        D = x.size(-1)
        B, H, W = segments.size()
        segments = segments.reshape([x.size(0), -1])-1 # batch, img_size^2


        batch_indices = torch.arange(x.size(0), device=x.device).unsqueeze(-1)  # (B, 1)
        spx_selected = x[batch_indices, segments]  # (B, H*W, D)

        # Reshape to (B, H, W, D)
        # x = spx_selected.view(B, H, W, D).permute(0, 3, 1, 2)

        # x = x.reshape(x.size(0), self.img_size, self.img_size, -1).permute(0, 3, 1, 2)
        x, _ = self.refinement(spx_selected)
        # x = x.reshape(x.size(0), 28, 28, 8, 8).permute(0, 1, 3, 2, 4).reshape(x.size(0), 1, H, W)
        x = x.reshape(x.size(0), H, W, -1).permute(0, 3, 1, 2)
        x = self.shallow(x)
        return intermediate, x
 

