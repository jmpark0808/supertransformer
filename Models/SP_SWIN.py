import torch.nn as nn
from Blocks.GraphBlocks import *
from Wrappers.PositionalEncoding import PositionalEncodingSuperPixel
from Blocks.swintransformer import *
import matplotlib.pyplot as plt
from dataset.constants import *

class SP_SWINU(nn.Module):
    '''
    Pure Global aggregation using transformers
    Deterministic Positional Encoding 
    '''
    def __init__(self, nfeat, nhid, nheads, ntfm, dropout, dropout_edge):
        """Dense version of GAT."""
        super().__init__()
        self.pos_linear = nn.Linear(2, nhid)
        options = {'swin_hp': {'patch_size': 1,  # (int | tuple(int)): Patch size. Default: 4
        'embed_dim': nhid, #(int): Patch embedding dimension. Default: 96
        'depths': [ntfm, ntfm, ntfm, ntfm], #(tuple(int)): Depth of each Swin Transformer layer.
        'num_heads': [nheads, nheads, nheads, nheads], #(tuple(int)): Number of attention heads in different layers.
        'window_size': 4, #(int): Window size. Default: 8
        'mlp_ratio': 2.,#(float): Ratio of mlp hidden dim to embedding dim. Default: 4
        'qkv_bias': True,#(bool): If True, add a learnable bias to query, key, value. Default: True
        'qk_scale': None,#(float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        'drop_rate': dropout,#(float): Dropout rate. Default: 0
        'attn_drop_rate': dropout_edge,#(float): Attention dropout rate. Default: 0
        'drop_path_rate': 0.1,#(float): Stochastic depth rate. Default: 0.1
        'norm_layer': nn.LayerNorm,#(nn.Module): Normalization layer. Default: nn.LayerNorm.
        'ape': False,#(bool): If True, add absolute position embedding to the patch embedding. Default: False
        'patch_norm': True,#(bool): If True, add normalization after patch embedding. Default: True
        'use_checkpoint': False,#(bool): Whether to use checkpointing to save memory. Default: False
        }, 
        'in_channels': nfeat,
        'patch_size': 32}
        self.model = SwinUTransformer(options = options)
        self.out = nn.Linear(128+64+32+16, 1)
    def forward(self, x):
        pos = x[:, :, :2]
        x = x[:, :, 2:]
       
        pos = self.pos_linear(pos)

        x = x.reshape(x.size(0), 32, 32, -1).permute(0, 3, 1, 2)
        x = self.model(x, pos)

        x = self.out(x)
        return x
    


class SP_SWIN(nn.Module):
    '''
    Pure Global aggregation using transformers
    Deterministic Positional Encoding 
    '''
    def __init__(self, nfeat, nhid, head_dim, nheads, ntfm, dropout, dropout_edge, kernels, window_size):
        """Dense version of GAT."""
        super().__init__()
        self.pos_linear = nn.Linear(2, nhid)
        options = {'swin_hp': {'patch_size': 1,  # (int | tuple(int)): Patch size. Default: 4
        'embed_dim': nhid, #(int): Patch embedding dimension. Default: 96
        'head_dim': head_dim,
        'depths': None, #(tuple(int)): Depth of each Swin Transformer layer.
        'num_heads': nheads, #(tuple(int)): Number of attention heads in different layers.
        'kernels': kernels,
        'window_size': window_size, #(int): Window size. Default: 8
        'mlp_ratio': 2.,#(float): Ratio of mlp hidden dim to embedding dim. Default: 4
        'qkv_bias': True,#(bool): If True, add a learnable bias to query, key, value. Default: True
        'qk_scale': None,#(float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        'drop_rate': dropout,#(float): Dropout rate. Default: 0
        'attn_drop_rate': dropout_edge,#(float): Attention dropout rate. Default: 0
        'drop_path_rate': 0.,#(float): Stochastic depth rate. Default: 0.1
        'norm_layer': nn.LayerNorm,#(nn.Module): Normalization layer. Default: nn.LayerNorm.
        'ape': False,#(bool): If True, add absolute position embedding to the patch embedding. Default: False
        'patch_norm': True,#(bool): If True, add normalization after patch embedding. Default: True
        'use_checkpoint': False,#(bool): Whether to use checkpointing to save memory. Default: False
        }, 
        'in_channels': nfeat,
        'patch_size': 32}
        self.model = SwinTransformer(options = options)
        self.out = nn.Linear(nhid, 1)
    def forward(self, x):
        pos = x[:, :, :2]
        x = x[:, :, 2:]
        
        pos = self.pos_linear(pos)
        x = x.reshape(x.size(0), 32, 32, -1).permute(0, 3, 1, 2)
        x = self.model(x, pos)

        x = self.out(x)
        return x
    
class SP_SWIN_Kernel(nn.Module):
    '''
    Pure Global aggregation using transformers
    Deterministic Positional Encoding 
    '''
    def __init__(self, nfeat, nhid, nheads, ntfm, dropout, dropout_edge, dilation):
        """Dense version of GAT."""
        super().__init__()
        self.pos_linear = nn.Linear(2, nhid)
        options = {'swin_hp': {'patch_size': 1,  # (int | tuple(int)): Patch size. Default: 4
        'embed_dim': nhid, #(int): Patch embedding dimension. Default: 96
        'depths': list(range(ntfm, 0, -1)), #(tuple(int)): Depth of each Swin Transformer layer.
        'num_heads': [nheads]*ntfm, #(tuple(int)): Number of attention heads in different layers.
        'window_size': dilation, #(int): Window size. Default: 8
        'mlp_ratio': 2.,#(float): Ratio of mlp hidden dim to embedding dim. Default: 4
        'qkv_bias': True,#(bool): If True, add a learnable bias to query, key, value. Default: True
        'qk_scale': None,#(float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        'drop_rate': dropout,#(float): Dropout rate. Default: 0
        'attn_drop_rate': dropout_edge,#(float): Attention dropout rate. Default: 0
        'drop_path_rate': 0.,#(float): Stochastic depth rate. Default: 0.1
        'norm_layer': nn.LayerNorm,#(nn.Module): Normalization layer. Default: nn.LayerNorm.
        'ape': False,#(bool): If True, add absolute position embedding to the patch embedding. Default: False
        'patch_norm': True,#(bool): If True, add normalization after patch embedding. Default: True
        'use_checkpoint': False,#(bool): Whether to use checkpointing to save memory. Default: False
        }, 
        'in_channels': nfeat,
        'patch_size': 32}
        self.model = SwinKernelTransformer(options = options)
        self.out = nn.Linear(nhid, 1)
    def forward(self, x):
        pos = x[:, :, :2]
        x = x[:, :, 2:]
       
        pos = self.pos_linear(pos)

        x = x.reshape(x.size(0), 32, 32, -1).permute(0, 3, 1, 2)
        x = self.model(x, pos)

        x = self.out(x)
        return x
    

class SP_SWIN_ImageNet(nn.Module):
    '''
    SWIN Transformer for ImageNet 
    '''
    def __init__(self, nfeat, nhid, head_dim, nheads, ntfm, dropout, dropout_edge, kernels, window_size):
        """Dense version of GAT."""
        super().__init__()
        self.pos_linear = nn.Linear(2, nhid)
        options = {'swin_hp': {'patch_size': 1,  # (int | tuple(int)): Patch size. Default: 4
        'embed_dim': nhid, #(int): Patch embedding dimension. Default: 96
        'head_dim': head_dim,
        'depths': None, #(tuple(int)): Depth of each Swin Transformer layer.
        'num_heads': nheads, #(tuple(int)): Number of attention heads in different layers.
        'kernels': kernels,
        'window_size': window_size, #(int): Window size. Default: 8
        'mlp_ratio': 2.,#(float): Ratio of mlp hidden dim to embedding dim. Default: 4
        'qkv_bias': True,#(bool): If True, add a learnable bias to query, key, value. Default: True
        'qk_scale': None,#(float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        'drop_rate': dropout,#(float): Dropout rate. Default: 0
        'attn_drop_rate': dropout_edge,#(float): Attention dropout rate. Default: 0
        'drop_path_rate': 0.,#(float): Stochastic depth rate. Default: 0.1
        'norm_layer': nn.LayerNorm,#(nn.Module): Normalization layer. Default: nn.LayerNorm.
        'ape': False,#(bool): If True, add absolute position embedding to the patch embedding. Default: False
        'patch_norm': True,#(bool): If True, add normalization after patch embedding. Default: True
        'use_checkpoint': False,#(bool): Whether to use checkpointing to save memory. Default: False
        }, 
        'in_channels': nfeat,
        'patch_size': 32}
        self.model = SwinTransformer(options = options)
        self.out = nn.Linear(nhid*16, 1000)
        self.unfold = torch.nn.Unfold(8, stride=8)
    def forward(self, x):
        pos = x[:, :, :2]
        x = x[:, :, 2:]
       
        pos = self.pos_linear(pos)

        x = x.reshape(x.size(0), 32, 32, -1).permute(0, 3, 1, 2)
        x = self.model(x, pos)
        x = x.reshape(x.size(0), 32, 32, -1).permute(0, 3, 1, 2) # B, C, 32, 32
        x = self.unfold(x) # B, C*8*8, 4*4
        x = x.reshape(x.size(0), -1, 8*8, 4*4)
        x = torch.mean(x, dim=2) # B, C, 4*4
        x = x.reshape(x.size(0), -1) # B, C*4*4
        x = self.out(x)
        return x