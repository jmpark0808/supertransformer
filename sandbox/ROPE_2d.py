from math import sqrt, pi, log

import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange


# class AxialRotaryEmbedding(nn.Module):
#     def __init__(self, dim, max_freq = 10):
#         super().__init__()
#         self.dim = dim
#         scales = torch.linspace(1., max_freq / 2, self.dim // 4)
#         self.register_buffer('scales', scales)

#     def forward(self, x): #xs, ys):
#         # xs: batch_size, seq_len, 1
#         # ys: batch_size, seq_len, 1


#         device, dtype, n = x.device, x.dtype, int(sqrt(x.shape[-2]))

#         seq = torch.linspace(-1., 1., steps = n, device = device)
#         seq = seq.unsqueeze(-1)
#         print(seq.size())

#         scales = self.scales[(*((None,) * (len(seq.shape) - 1)), Ellipsis)]
#         scales = scales.to(device)

#         seq = seq * scales * pi
#         print(seq.size())

#         x_sinu = repeat(seq, 'i d -> i j d', j = n)
#         y_sinu = repeat(seq, 'j d -> i j d', i = n)
#         print(x_sinu.size())

#         sin = torch.cat((x_sinu.sin(), y_sinu.sin()), dim = -1)
#         cos = torch.cat((x_sinu.cos(), y_sinu.cos()), dim = -1)
#         print(sin.size())

#         sin, cos = map(lambda t: rearrange(t, 'i j d -> (i j) d'), (sin, cos))
#         print(sin.size())
#         sin, cos = map(lambda t: repeat(t, 'n d -> () n (d j)', j = 2), (sin, cos))
#         print(sin.size())
#         return sin, cos
    


    

ar = AxialRotaryEmbedding(64)
x = torch.randn(10, 625, 1)
result = ar(x, x)
print(result[0].size(), result[1].size())