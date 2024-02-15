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
        self.gunet = GraphUNet(nhid, nhid, 1, depth=ntfm, pool_ratios=0.75, dropout=dropout, heads=nheads, sum_res=False)

        self.num_seg = num_seg
        
    def forward(self, data, return_perms=False):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        pos = x[:, :2]
        x = x[:, 2:]

        x = self.linear1(x)
        batch_size = x.size(0)//self.num_seg
        batch_index = torch.arange(0, batch_size).repeat(self.num_seg).reshape(self.num_seg, -1).T.reshape(-1).cuda()
        pos = self.pos_linear(pos)
        x += pos

        x, perms, edge_indices = self.gunet(x, edge_index, batch_index)
        if return_perms:
            return x, perms, edge_indices
        return x