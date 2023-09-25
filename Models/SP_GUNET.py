import torch.nn as nn
from Blocks.GraphBlocks import *
from Blocks.TransformerBlocks import *
from dataset.constants import *
from Blocks.CustomGUNet import GraphUNet


class SP_GUNET_PyG(nn.Module):
    '''
    Pure Global aggregation using transformers
    Deterministic Positional Encoding 
    '''
    def __init__(self, nfeat, nhid, edge_dim, dropout, nheads, ntfm):
        """Dense version of GAT."""
        super(SP_GUNET_PyG, self).__init__()
        self.unet = GraphUNet(nfeat, nhid, 1, depth=ntfm, heads=nheads, pool_ratios=0.8, dropout=dropout, edge_dim=edge_dim)

        
    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        x = self.unet(x, edge_index, edge_attr)
        return x