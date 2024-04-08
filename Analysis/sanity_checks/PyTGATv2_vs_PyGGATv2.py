from torch_geometric.nn import GATv2Conv, GATConv
from Blocks.GraphBlocks import GATv3, GATv2
import torch
import torch.nn as nn
from torch_geometric.nn.dense.linear import Linear
torch.manual_seed(0)
dummy_input = torch.randn(5, 3)

edge_index = torch.ones(5, 5)
edge_index = edge_index.nonzero().t().contiguous()

pyg = GATv2Conv(3, 3, 1, False, 0.2, 0)
pyt = GATv3(3, 3, 0, 1, 0.2, False)

# def init_weights(m):
#     if isinstance(m, nn.Linear):
#         m.weight.data.fill_(0.5)
#         if m.bias is not None:
#             m.bias.data.fill_(0.1)
#     if isinstance(m, Linear):
#         m.weight.data.fill_(0.5)
#         if m.bias is not None:
#             m.bias.data.fill_(0.1)
    
# pyg.apply(init_weights)
# pyt.apply(init_weights)

lin_r_weights = torch.randn(3, 3)
lin_l_weights = torch.randn(3, 3)
lin_r_bias = torch.randn(3)
lin_l_bias = torch.randn(3)


with torch.no_grad():
    pyg.lin_l.weight.copy_(lin_l_weights)
    pyg.lin_l.bias.copy_(lin_l_bias)
    pyg.lin_r.weight.copy_(lin_r_weights)
    pyg.lin_r.bias.copy_(lin_r_bias)

    pyt.lin_l.weight.copy_(lin_l_weights)
    pyt.lin_l.bias.copy_(lin_l_bias)
    pyt.lin_r.weight.copy_(lin_r_weights)
    pyt.lin_r.bias.copy_(lin_r_bias)

att = torch.randn(1, 1, 3)
pyg.att= nn.Parameter(att)
pyt.att = nn.Parameter(att.unsqueeze(-1).unsqueeze(-1))

# pyg.att_src = nn.Parameter(torch.ones(1, 1, 3))
# pyg.att_dst = nn.Parameter(torch.ones(1, 1, 3))
# pyt.lin_l = nn.Parameter(torch.ones(1, 1, 1, 3))
# pyt.lin_r = nn.Parameter(torch.ones(1, 1, 1, 3))


pyg_output = pyg(dummy_input, edge_index)
print(pyg_output)
pyt_output = pyt(dummy_input.unsqueeze(0), None)
print(pyt_output)

