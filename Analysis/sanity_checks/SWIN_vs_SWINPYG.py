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




def init_weights(m):
    if isinstance(m, nn.Linear):
        m.weight.data.fill_(0.5)
        if m.bias is not None:
            m.bias.data.fill_(0.1)
    if isinstance(m, nn.Conv2d):
        m.weight.data.fill_(0.5)
        if m.bias is not None:
            m.bias.data.fill_(0.1)
    if isinstance(m, Linear):
        m.weight.data.fill_(0.5)
        if m.bias is not None:
            m.bias.data.fill_(0.1)
    if isinstance(m, nn.LayerNorm):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)
    if isinstance(m, LayerNorm):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)

pyg_swin = SP_SWIN_PyG(3, 5, 8, None, 0, 8, 6, 1024, 4).cuda()
print(pyg_swin)
pyg_swin.apply(init_weights)
pyg_swin.eval()


swin = SP_SWIN(3, 5, 8, 8, 6, 0, 0, [4, 4, 4], 4, 320).cuda()
print(swin)
swin.apply(init_weights)
swin.eval()

window_size = 4
node_index = torch.arange(1024).reshape(1, 32, 32).float()
local_unfold = torch.nn.Unfold(window_size, stride=window_size)
dilated_unfold = torch.nn.Unfold(window_size, dilation=(32//window_size))


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
    x, y = torch.meshgrid(dilated_index[:, i], dilated_index[:, i])
    all_dilated_indices.append(torch.stack((x, y), dim=0).reshape(2, -1))
    
all_local_indices = torch.cat(all_local_indices, dim=1)
all_shifted_indices = torch.cat(all_shifted_indices, dim=1)
all_dilated_indices = torch.cat(all_dilated_indices, dim=1)

edge_indices = torch.cat((all_local_indices, all_shifted_indices, all_dilated_indices), dim=1).long()



x = torch.randn(1024, 5)

pyg_x = Data(x=x, edge_index=edge_indices, edge_attr=torch.zeros_like(edge_indices)).cuda() 

out_pyg = pyg_swin(pyg_x)

print(summary(pyg_swin, pyg_x, max_depth=8))
# print(flop_count_table(flops))

# flops = FlopCountAnalysis(pyt_tfm, (x, None, torch.tensor(neighbor_array_pyt), None))
out_pyt = swin(x.cuda().unsqueeze(0))
from torchsummary import summary

print(summary(swin, (1024, 5)))
# print(flop_count_table(flops))

# print(torch.sum(torch.abs(out_pyg-out_pyt)))


          