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
    
# class AxialRotaryEmbedding(nn.Module):
#     def __init__(self, dim, max_freq = 10):
#         super().__init__()
#         self.dim = dim
#         scales = torch.linspace(1., max_freq / 2, self.dim // 4)
#         self.register_buffer('scales', scales)

#     def forward(self, xs, ys):
#         # xs: batch_size, seq_len, 1
#         # ys: batch_size, seq_len, 1
#         device, dtype = xs.device, xs.dtype


#         scales = self.scales[(*((None,) * (len(xs.shape) - 1)), Ellipsis)]
#         scales = scales.to(device)

#         seq_x = xs * scales * math.pi # N x d//4
#         seq_y = ys * scales * math.pi # N x d//4

#         sin = torch.cat((seq_x.sin(), seq_y.sin()), dim = -1) # N x d//2
#         cos = torch.cat((seq_x.cos(), seq_y.cos()), dim = -1) # N x d//2

#         sin, cos = map(lambda t: repeat(t, 'b n d -> b n (d j)', j = 2), (sin, cos))
#         return sin, cos
    
class AxialRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_freq = 10):
        super().__init__()
        self.dim = dim
        scales = torch.linspace(1., max_freq / 2, self.dim // 4)
        self.d = torch.arange(1, self.dim//4+1, device='cuda')
        self.register_buffer('scales', scales)

    def forward(self, xs, ys):
        

        # seq = torch.linspace(-1., 1., steps = n, device = device)
        # seq = seq.unsqueeze(-1)

        # scales = self.scales[(*((None,) * (len(seq.shape) - 1)), Ellipsis)]
        # scales = scales.to(x)
        seq_x = xs
        seq_y = ys



        theta = (10000**(-2.*(self.d-1)/self.dim))
        theta = theta.unsqueeze(0).unsqueeze(0)

        seq_x = seq_x * theta # N^2 x d//4
        seq_y = seq_y * theta # N^2 x d//4


        # x_sinu = repeat(seq, 'i d -> i j d', j = n)
        # y_sinu = repeat(seq, 'j d -> i j d', i = n)

        sin = torch.cat((seq_x.sin(), seq_y.sin()), dim = -1) # N^2 x d//2
        cos = torch.cat((seq_x.cos(), seq_y.cos()), dim = -1) # N^2 x d//2

        sin, cos = map(lambda t: repeat(t, 'b n d -> b n (d j)', j = 2), (sin, cos))
        return sin, cos


class SP_SWIN(nn.Module):
    '''
    Pure Global aggregation using transformers
    Deterministic Positional Encoding 
    '''
    def __init__(self, nfeat, nhid, head_dim, nheads, ntfm, dropout, dropout_edge, kernels, window_size, image_size):
        """Dense version of GAT."""
        super().__init__()
        
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
        self.pos_linear = nn.Linear(2, nhid)
        # self.pos_linear_y = nn.Linear(1, nhid//2)
        self.model = SwinTransformer(options = options)
        self.out = nn.Linear(nhid, 1)
        # self.pos_emb =  AxialRotaryEmbedding(head_dim, max_freq=image_size)
        self.image_size = image_size
    def forward(self, x):
        pos = x[:, :, :2]
        # rel_pos = torch.sqrt(torch.sum(torch.pow(pos.unsqueeze(2) - pos.unsqueeze(1), 2), dim=-1)) # B, N, N
        x = x[:, :, 2:]
        
        # pos_y = self.pos_linear_y(pos_y)
        # pos_x = self.pos_linear_x(pos_x)
        pos_emb = self.pos_linear(pos)
        
        # pos = pos.reshape(pos.size(0), 32, 32, -1)
        x = x.reshape(x.size(0), 32, 32, -1).permute(0, 3, 1, 2)
        x = self.model(x, pos_emb)

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
        self.out = nn.Linear(nhid, 1000)
       
    def forward(self, x):
        pos = x[:, :, :2]
        x = x[:, :, 2:]
       
        pos = self.pos_linear(pos)

        x = x.reshape(x.size(0), 32, 32, -1).permute(0, 3, 1, 2)
        x = self.model(x, pos)
        x = torch.mean(x, dim=1) 
        x = self.out(x)
        return x