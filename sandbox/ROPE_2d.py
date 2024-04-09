from math import sqrt, pi, log

import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import math

class AxialRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_freq = 10):
        super().__init__()
        self.dim = dim
        scales = torch.linspace(1., max_freq / 2, self.dim // 4)
        self.register_buffer('scales', scales)

    def forward(self, xs, ys):
        # xs: batch_size, seq_len, 1
        # ys: batch_size, seq_len, 1
        device, dtype = xs.device, xs.dtype


        scales = self.scales[(*((None,) * (len(xs.shape) - 1)), Ellipsis)]
        scales = scales.to(device)
       
        seq_x = xs * scales * math.pi # B x N^2 x d//4
        seq_y = ys * scales * math.pi # B x N^2 x d//4


        sin = torch.cat((seq_x.sin(), seq_y.sin()), dim = -1) # B x N^2 x d//2
        cos = torch.cat((seq_x.cos(), seq_y.cos()), dim = -1) # B x N^2 x d//2
 

        sin, cos = map(lambda t: repeat(t, 'b n d -> b n (d j)', j = 2), (sin, cos))
        return sin, cos
    
# class AxialRotaryEmbedding(nn.Module):
#     def __init__(self, dim, max_freq = 10):
#         super().__init__()
#         self.dim = dim
#         scales = torch.linspace(1., max_freq / 2, self.dim // 4)
#         self.register_buffer('scales', scales)

#     def forward(self, x):
#         device, dtype, n = x.device, x.dtype, int(sqrt(x.shape[-2]))

#         seq = torch.linspace(-1., 1., steps = n, device = device)
#         seq = seq.unsqueeze(-1)

#         scales = self.scales[(*((None,) * (len(seq.shape) - 1)), Ellipsis)]
#         scales = scales.to(x)

#         seq = seq * scales * pi

#         x_sinu = repeat(seq, 'i d -> i j d', j = n) # N x N x d//4
#         y_sinu = repeat(seq, 'j d -> i j d', i = n) # N x N x d//4

#         sin = torch.cat((x_sinu.sin(), y_sinu.sin()), dim = -1) # N x N x d//2
#         cos = torch.cat((x_sinu.cos(), y_sinu.cos()), dim = -1) # N x N x d//2

#         sin, cos = map(lambda t: rearrange(t, 'i j d -> (i j) d'), (sin, cos)) # N^2 x d//2
#         sin, cos = map(lambda t: repeat(t, 'n d -> () n (d j)', j = 2), (sin, cos)) # N^2 x d
#         return sin, cos


    

ar = AxialRotaryEmbedding(64)
# x = torch.arange(1, 26).unsqueeze(-1).float()
x = torch.randn(16, 625, 64)
result = ar(x)
print(result[0].size(), result[1].size())
