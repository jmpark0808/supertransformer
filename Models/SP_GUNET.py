import torch.nn as nn
from Blocks.GraphBlocks import *
from Blocks.TransformerBlocks import *
from dataset.constants import *
from Blocks.CustomGUNet import GraphUNet
from Blocks.CustomGUNet_graclus import GraphUNet as GraphUNet_G
from torch_geometric.utils import dropout_edge


class SP_GUNET_PyG(nn.Module):
    '''
    Pure Global aggregation using transformers
    Deterministic Positional Encoding 
    '''
    def __init__(self, nfeat, nhid, nheads, num_seg, ntfm, dropout, mode):
        """Dense version of GAT."""
        super(SP_GUNET_PyG, self).__init__()
        self.linear1 = nn.Linear(nfeat, nhid)
        self.pos_linear = nn.Linear(2, nhid)
        if mode == 'graclus':
            self.gunet = GraphUNet_G(nhid, nhid, 1, depth_pool=ntfm, depth_conv=ntfm, pool_ratios=0.75, dropout=dropout, heads=nheads, sum_res=False)
        else:
            self.gunet = GraphUNet(nhid, nhid, 1, depth=ntfm, pool_ratios=0.75, dropout=dropout, heads=nheads, sum_res=False)

        self.num_seg = num_seg
        
    def forward(self, data, return_perms=False):
       
        data.pos = data.x[:, :2]
        data.x = data.x[:, 2:]

        data.x = self.linear1(data.x)
        batch_size = data.x.size(0)//self.num_seg
        batch_index = torch.arange(0, batch_size).repeat(self.num_seg).reshape(self.num_seg, -1).T.reshape(-1).cuda()
        pos = self.pos_linear(data.pos)
        data.x += pos

        x, perms, edge_indices = self.gunet(data, batch_index)
        if return_perms:
            return x, perms, edge_indices
        return x