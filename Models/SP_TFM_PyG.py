import torch.nn as nn
from Blocks.GraphBlocks import *
from Blocks.TransformerBlocks import *
from dataset.constants import *
from torch_geometric.nn.conv import TransformerConv
from Blocks.TransformerBlocks import FeedForward


class SP_TFM_PyG(nn.Module):
    '''
    Pure Global aggregation using transformers
    Deterministic Positional Encoding 
    '''
    def __init__(self, nfeat, nhid, edge_dim, dropout, nheads, ntfm):
        """Dense version of GAT."""
        super(SP_TFM_PyG, self).__init__()
        self.linear1 = nn.Linear(nfeat, nhid)
        self.elu = nn.ELU()
    
        self.convs = nn.ModuleList([TransformerConv(in_channels=nhid, out_channels=nhid, heads=nheads, dropout=dropout, edge_dim=edge_dim, concat=False) for _ in range(ntfm)])
        self.ff = nn.ModuleList([FeedForward(nhid, nhid*2, dropout) for _ in range(ntfm)])
        self.classifier = nn.Linear(nhid, 1)


        
    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        x = self.linear1(x)
        x = self.elu(x)

        for conv, ff in zip(self.convs, self.ff):
            x = conv(x, edge_index, edge_attr=edge_attr) # adding edge features here!
            x = ff(x)
      
        # x = self.convs[-1](x, edge_index, edge_attr=edge_attr)
        x = self.classifier(x)
        return x