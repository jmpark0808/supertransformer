import torch.nn as nn

import torch
import sys
sys.path.insert(0, '/home/eddie/waterloo/supertransformer')
import numpy as np
from torch_geometric.data import Data

from fvcore.nn import FlopCountAnalysis
from fvcore.nn import flop_count_table
from einops import rearrange

from Models.SP_SWIN import SP_SWIN
from Models.SP_SWIN_PyG import SP_SWIN_PyG


import math
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptTensor, PairTensor, SparseTensor
from torch_geometric.utils import softmax
from torch_geometric.loader import DataLoader
from torch_geometric.nn.norm import LayerNorm
# from torchsummary import summary
from torch_geometric.nn import summary


d_w = {}

torch.manual_seed(22)
def init_weights(m):
    if isinstance(m, nn.Linear):
        if (m.in_features, m.out_features) not in d_w:
            nn.init.normal_(m.weight)
            d_w[(m.in_features, m.out_features)] = m.weight
            if m.bias is not None:
                m.bias.data.fill_(0)
        else:
            m.weight.data = d_w[(m.in_features, m.out_features)]
            if m.bias is not None:
                m.bias.data.fill_(0)
    if isinstance(m, nn.Conv2d):
        m.weight.data = d_w[(3, 5)].unsqueeze(2).unsqueeze(3)
        if m.bias is not None:
            m.bias.data.fill_(0)
    if isinstance(m, Linear):
        if (m.in_channels, m.out_channels) not in d_w:
            nn.init.normal_(m.weight)
            d_w[(m.in_channels, m.out_channels)] = m.weight
            if m.bias is not None:
                m.bias.data.fill_(0)
        else:
            m.weight.data = d_w[(m.in_channels, m.out_channels)]
            if m.bias is not None:
                m.bias.data.fill_(0)
    if isinstance(m, nn.LayerNorm):
        if m.normalized_shape not in d_w:
            nn.init.normal_(m.weight)
            d_w[(m.normalized_shape)] = m.weight
            if m.bias is not None:
                m.bias.data.fill_(0)
        else:
            m.weight.data = d_w[m.normalized_shape]
            if m.bias is not None:
                m.bias.data.fill_(0)
    if isinstance(m, LayerNorm):
        if (m.in_channels,) not in d_w:
            nn.init.normal_(m.weight)
            d_w[(m.in_channels,)] = m.weight
            if m.bias is not None:
                m.bias.data.fill_(0)
        else:
            m.weight.data = d_w[(m.in_channels,)]
            if m.bias is not None:
                m.bias.data.fill_(0)


pyg_swin = SP_SWIN_PyG(3, 5, 8, None, 0, 8, 6, 1024, 4).cuda()

pyg_swin.apply(init_weights)
pyg_swin.eval()

swin = SP_SWIN(3, 5, 8, 8, 6, 0, 0, [4, 4, 4], 4, 320).cuda()

swin.apply(init_weights)
swin.eval()


window_size = 4
node_index = torch.arange(1024).reshape(1, 32, 32).float()
local_unfold = torch.nn.Unfold(window_size, stride=window_size)
dilated_unfold = torch.nn.Unfold((32//window_size), dilation=window_size)


local_index = local_unfold(node_index)
shifted_index = local_unfold(torch.roll(node_index, shifts=(-window_size//2, -window_size//2), dims=(1, 2)))
dilated_index = dilated_unfold(node_index)


all_local_indices = []
all_shifted_indices = []
all_dilated_indices = []
for i in range(local_index.shape[1]):
    x, y = torch.meshgrid(local_index[:, i], local_index[:, i])
    all_local_indices.append(torch.stack((x, y), dim=0).reshape(2, -1))
    x, y = torch.meshgrid(shifted_index[:, i], shifted_index[:, i])
    all_shifted_indices.append(torch.stack((x, y), dim=0).reshape(2, -1))

for i in range(dilated_index.shape[1]):
    x, y = torch.meshgrid(dilated_index[:, i], dilated_index[:, i])
    all_dilated_indices.append(torch.stack((x, y), dim=0).reshape(2, -1))
    
all_local_indices = torch.cat(all_local_indices, dim=1)
all_shifted_indices = torch.cat(all_shifted_indices, dim=1)
all_dilated_indices = torch.cat(all_dilated_indices, dim=1)

edge_indices = torch.cat((all_local_indices, all_shifted_indices, all_dilated_indices), dim=1).long()

# weight = nn.Parameter(torch.tensor(np.random.normal(0, 1, 5).astype(float)))
# layer_norm_pyt = nn.LayerNorm(5)
# layer_norm_pyg = LayerNorm(5, mode='node')
# layer_norm_pyt.weight = weight
# layer_norm_pyt.bias = nn.Parameter(torch.tensor(np.zeros(5).astype(float)))
# layer_norm_pyg.weight = weight
# layer_norm_pyg.bias = nn.Parameter(torch.tensor(np.zeros(5).astype(float)))

x = torch.randn(1024, 5)
x_coordinates = torch.arange(0, 320, 10)
y_coordinates = torch.arange(0, 320, 10)
x_, y_ = torch.meshgrid(x_coordinates, y_coordinates)
coordinates = torch.stack((x_, y_), dim=2).reshape(1024, 2)
x[:, :2] = coordinates.type(x.dtype)


# ln_pyt = layer_norm_pyt(x)
# ln_pyg = layer_norm_pyg(x)
# print(ln_pyt)
# print(ln_pyg)
# assert(0)

pyg_x = Data(x=x, edge_index=edge_indices, edge_attr=torch.zeros_like(edge_indices)).cuda() 


out_pyg = pyg_swin(pyg_x)


print(out_pyg)
# print(flop_count_table(flops))

# flops = FlopCountAnalysis(pyt_tfm, (x, None, torch.tensor(neighbor_array_pyt), None))

out_pyt = swin(x.cuda().unsqueeze(0))

from torchsummary import summary

print(out_pyt)
# print(flop_count_table(flops))

# print(torch.sum(torch.abs(out_pyg-out_pyt)))


# a = torch.tensor([ 0.5540,  0.2412,  0.8444, -0.7500, -0.9816,  1.7255, -1.9035, -0.6444])
# b = torch.tensor([-0.0702,  0.0143, -0.2342,  0.5122,  0.5286, -0.6572, -1.1298, -1.8607])
# print(a@b)