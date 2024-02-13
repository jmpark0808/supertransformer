import torch.nn as nn
from Blocks.GraphBlocks import *
from Blocks.TransformerBlocks import *
from dataset.constants import *
from Blocks.CustomGUNet import GraphUNet
from torch_geometric.utils import dropout_edge


class SP_GUNET_PyG(nn.Module):
    '''
    Pure Global aggregation using transformers
    Deterministic Positional Encoding 
    '''
    def __init__(self, nfeat, nhid, nheads, num_seg, ntfm, dropout):
        """Dense version of GAT."""
        super(SP_GUNET_PyG, self).__init__()
        self.linear1 = nn.Linear(nfeat, nhid)
        self.pos_linear = nn.Linear(2, nhid)
        self.gunet = GraphUNet(nhid, nhid, 1, depth=ntfm, pool_ratios=0.5, dropout=dropout, heads=nheads)

        self.num_seg = num_seg
        
    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        pos = x[:, :2]
        x = x[:, 2:]

        x = self.linear1(x)

        pos = self.pos_linear(pos)
        x += pos

        x = self.gunet(x, edge_index)
        return x