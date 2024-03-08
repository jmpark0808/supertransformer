from torch_geometric.nn.conv import TransformerConv
import torch

layer = TransformerConv(36, 36, 8, False).cuda()
input = torch.randn([625, 36]).cuda()
edge = torch.ones(625, 625).cuda()
edge = edge.nonzero().t().contiguous()

layer(input, edge)