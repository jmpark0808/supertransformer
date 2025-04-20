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
from torch.nn import init
import math
from einops import rearrange, repeat
from Blocks.swin_common import PatchEmbed
from Blocks.upernet import  SwinASPP, SwinDecoder
from Blocks.swin_rpe import BasicLayerRPE, PatchMergingRPE
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
        fused_window_process (bool, optional): If True, use one kernel to fused window shift & window partition for acceleration, similar for the reversed part. Default: False
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=[96, 192, 384], depths=[2, 2, 6], num_heads=[3, 6, 12],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, fused_window_process=False, factor=1.0, **kwargs):
        super().__init__()
        assert len(depths) == len(num_heads) == len(embed_dim), 'Number of layers must be equal between heads, dims, and depths'
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim[0]
        self.ape = ape
        self.patch_norm = patch_norm
        # self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim[0],
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding


        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        dim_list = []
        resolution_list = []
        for i_layer in range(self.num_layers):
            layer = BasicLayerRPE(dim=embed_dim[i_layer],
                                  out_dim=embed_dim[i_layer+1] if i_layer < self.num_layers-1 else None,
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=0,
                               norm_layer=norm_layer,
                               downsample=PatchMergingRPE if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint,
                               fused_window_process=fused_window_process)
            self.layers.append(layer)
            dim_list.append(embed_dim[i_layer])
            resolution_list.append((patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)))
        
        dim_list.reverse()
        resolution_list.reverse()
        
        
        self.upsample_layers = SwinDecoder(input_dim=embed_dim[0],# 输入的通道数为96
            input_high_dim = dim_list[0], # 384
            input_middle_dim = dim_list[1],
            input_size=resolution_list[0][0], # 14 × 14
            low_level_idx=0, # 0
            high_level_idx=2, # 2
            num_classes=1,
            depth=2, # 2
            last_layer_depth=6, # 6
            num_heads=num_heads[0], # 3
            window_size=window_size, # 7
            mlp_ratio=self.mlp_ratio, # 4
            qk_scale=qk_scale,
            qkv_bias=qkv_bias,
            drop_path_rate=drop_path_rate,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            norm_layer=norm_layer,
            decoder_norm=True, # True
            use_checkpoint=False)


        self.aspp = SwinASPP(
            input_size=resolution_list[0][0], # 14×14
            input_dim=dim_list[0], # 384 
            out_dim=dim_list[-1],  # 96
            depth=2, # 2
            cross_attn='CBAM', # CBAM
            num_heads=num_heads[0], # 3头
            mlp_ratio=self.mlp_ratio, # 4
            qk_scale=qk_scale,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=0, # 0.1
            norm_layer=norm_layer,
            aspp_norm=False,
            aspp_activation='relu', # relu
            start_window_size=2,
            aspp_dropout=0.1, # 0.1
            downsample=None, #None
            use_checkpoint=False
        )
        

        self.apply(self._init_weights)

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
        centroids = x[:, :2, :, : ]
        x = x[:, 2:, :, :]
        x = self.patch_embed(x)

        x = self.pos_drop(x)
  
        all_layers = []
        for idx, layer in enumerate(self.layers):
            pre_ds, x, centroids  = layer(x, centroids)
            size = int(math.sqrt(pre_ds.size(1)))
            all_layers.append(pre_ds.view(-1, size, size, pre_ds.shape[-1]))

        
        x = self.aspp(all_layers[-1])
        x = self.upsample_layers(all_layers[0], all_layers[1], all_layers[2], x)

       
        return x

    def forward(self, x):
        x = self.forward_features(x)
        
        
        return x


    


    
    




