import torch.nn as nn
from Blocks.GraphBlocks import *
from Blocks.TransformerBlocks import *
from dataset.constants import *
from torch_geometric.nn.models import GraphUNet
from torch_geometric.utils import dropout_edge


class SP_GUNET_PyG(nn.Module):
    '''
    Pure Global aggregation using transformers
    Deterministic Positional Encoding 
    '''
    def __init__(self, nfeat, nhid, num_seg, ntfm):
        """Dense version of GAT."""
        super(SP_GUNET_PyG, self).__init__()
        self.linear1 = nn.Linear(nfeat, nhid)
        self.elu = nn.ELU()
        self.pos_linear = nn.Linear(2, nhid)
        self.gunet = GraphUNet(nhid, nhid, 1, depth=ntfm, pool_ratios=0.8)

        self.num_seg = num_seg
        
    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        batch_size = x.size(0)//self.num_seg
        batch_index = torch.arange(0, batch_size).repeat(self.num_seg).reshape(self.num_seg, -1).T.reshape(-1).cuda()
        pos = x[:, :2]
        x = x[:, 2:]

        x = self.linear1(x)
        x = self.elu(x)

        pos = self.pos_linear(pos)
        x += pos

        x = self.gunet(x, edge_index, batch_index)
        return x