import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
from Blocks.swin_common import PatchEmbed
from Blocks.swin_rope import BasicLayerRoPE, PatchMergingRoPE
from util.util import TokenDropout

class SwinTransformer(nn.Module):
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
                 embed_dim=[96, 96*2, 96*4, 96*8], depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, fused_window_process=False, **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim[0]
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim[-1]
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches

        self.patch_embed_colour = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=6, embed_dim=embed_dim[0],
            norm_layer= None) #norm_layer if self.patch_norm else
        self.patch_embed_lbp = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=10, embed_dim=embed_dim[0],
            norm_layer= None) #norm_layer if self.patch_norm else
        self.patch_embed_fft = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans-24, embed_dim=embed_dim[0],
            norm_layer= None) #norm_layer if self.patch_norm else
        self.patch_embed_moments = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=8, embed_dim=embed_dim[0],
            norm_layer= None) #norm_layer if self.patch_norm else
        self.linear_embed = nn.Sequential(nn.Linear(embed_dim[0]*4, embed_dim[0]), nn.LayerNorm(embed_dim[0]), nn.ReLU(), nn.Linear(embed_dim[0], embed_dim[0]))
        num_patches = self.patch_embed_colour.num_patches
        patches_resolution = self.patch_embed_colour.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
 

        # self.pos_drop = nn.Dropout(p=drop_rate)
        self.pos_drop = TokenDropout(drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        self.resolutions = []
        self.embed_dims = []
        for i_layer in range(self.num_layers):
            layer = BasicLayerRoPE(dim=embed_dim[i_layer],
                                  out_dim=embed_dim[i_layer+1] if i_layer < self.num_layers-1 else None,
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMergingRoPE if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint,
                               fused_window_process=fused_window_process)
            resolution = (patches_resolution[0]// (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer))
            self.resolutions.append(resolution)
            self.embed_dims.append(embed_dim[i_layer])
            self.layers.append(layer)

        # self.locations = nn.Sequential(*[nn.Linear(img_size[0]*img_size[1], embed_dim[0])])
        self.locations = nn.Sequential(*[nn.Linear(2, embed_dim[0])])

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

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

    def forward_features(self, x, locations):
        colour = x[:, :6, :, :]
        fft = x[:, 6:-18, :, :]
        moments = x[:, -18:-10, :, :]
        lbp = x[:, -10:, :, :]

        
        colour_skip = colour
        fft_skip = fft
        moments_skip = moments
        lbp_skip = lbp

        colour = self.patch_embed_colour(colour)
        fft = self.patch_embed_fft(fft)
        moments = self.patch_embed_moments(moments)
        lbp = self.patch_embed_lbp(lbp)

        if torch.sum(torch.isinf(colour_skip)) > 0:
            print('Colour features Nan')
            nan_mask = torch.isinf(colour_skip)
            
            has_nan_along_A = nan_mask.any(dim=1)  # shape: (B, C, D)

            # Iterate through (b, c, d) where NaNs were found
            indices = torch.nonzero(has_nan_along_A, as_tuple=False)  # shape: (num_nans, 3)

            for b, c, d in indices:
                vector = colour_skip[b, :, c, d]  # shape: (A,)
                print('Raw features', vector)
            
           
            
            assert(0)
        if torch.sum(torch.isinf(fft_skip)) > 0:
            print('FFT features Nan')
            nan_mask = torch.isinf(fft_skip)
            
            has_nan_along_A = nan_mask.any(dim=1)  # shape: (B, C, D)

            # Iterate through (b, c, d) where NaNs were found
            indices = torch.nonzero(has_nan_along_A, as_tuple=False)  # shape: (num_nans, 3)

            for b, c, d in indices:
                vector = fft_skip[b, :, c, d]  # shape: (A,)
                print('Raw features', vector)
            assert(0)

        if torch.sum(torch.isinf(moments_skip)) > 0:
            print('Moments features Nan')
            nan_mask = torch.isinf(moments_skip)
            
            has_nan_along_A = nan_mask.any(dim=1)  # shape: (B, C, D)

            # Iterate through (b, c, d) where NaNs were found
            indices = torch.nonzero(has_nan_along_A, as_tuple=False)  # shape: (num_nans, 3)

            for b, c, d in indices:
                vector = moments_skip[b, :, c, d]  # shape: (A,)
                print('Raw features', vector)
            assert(0)

        if torch.sum(torch.isinf(lbp_skip)) > 0:
            print('LBP features Nan')
            nan_mask = torch.isinf(lbp_skip)
            
            has_nan_along_A = nan_mask.any(dim=1)  # shape: (B, C, D)

            # Iterate through (b, c, d) where NaNs were found
            indices = torch.nonzero(has_nan_along_A, as_tuple=False)  # shape: (num_nans, 3)

            for b, c, d in indices:
                vector = lbp_skip[b, :, c, d]  # shape: (A,)
                print('Raw features', vector)
            assert(0)

        features = torch.cat((colour, fft, moments, lbp), dim=-1)
        features_skip = features
        
        x = self.linear_embed(features)

        if torch.sum(torch.isnan(x)) > 0:
            print('Features Nan')
            nan_mask = torch.isnan(x)
            
            has_nan_along_A = nan_mask.any(dim=2)  # shape: (B, C)

            # Iterate through (b, c, d) where NaNs were found
            indices = torch.nonzero(has_nan_along_A, as_tuple=False)  # shape: (num_nans, 3)

            for b, c in indices:
                vector = x[b, c, :]  # shape: (A,)
                vector_skip = features_skip[b, c, :]
                print('Raw features', vector_skip)
                print('Model features', vector)

            
          
            assert(0)
        x = self.pos_drop(x)
        x = x + locations

        for layer in self.layers:
            ds, x = layer(x)
        if torch.sum(torch.isnan(x)) > 0:
            print('Backbone Nan')
            assert(0)
        x = self.norm(x)  # B L C
        if torch.sum(torch.isnan(x)) > 0:
            print('Norm Nan')
            assert(0)
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        centroids = x[:, :2, :, :]
        features = x[:, 2:, :, :]
        locations = centroids.permute(0, 2, 3, 1)
        locations = self.locations(locations)
        locations = locations.reshape(locations.size(0), -1, locations.size(3))
        
        x = self.forward_features(features, locations)
        x = self.head(x)
        return x