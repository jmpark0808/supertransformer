import torch.nn as nn
from Blocks.GraphBlocks import *
from Wrappers.PositionalEncoding import PositionalEncodingSuperPixel
from Blocks.TransformerBlocks import *
from dataset.constants import *
from torch_geometric.nn.conv import GATv2Conv


class SP_GAT_PyG(nn.Module):
    '''
    Pure Global aggregation using transformers
    Deterministic Positional Encoding 
    '''
    def __init__(self, nfeat, nhid, edge_dim, dropout, nheads, ntfm):
        """Dense version of GAT."""
        super(SP_GAT_PyG, self).__init__()
        self.linear1 = nn.Linear(nfeat, nhid)
        self.elu = nn.ELU()
    
        self.first_conv = GATv2Conv(in_channels=nhid, out_channels=nhid, heads=nheads, dropout=dropout, edge_dim=edge_dim)
        self.convs = [GATv2Conv(in_channels=nhid*nheads, out_channels=nhid, heads=nheads, dropout=dropout, edge_dim=edge_dim) for _ in range(ntfm)]
        self.classifier = nn.Linear(nhid, 1)


        
    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        x = self.linear1(x)
        x = self.elu(x)

        x = self.first_conv(x, edge_index, edge_attr=edge_attr)
        x = self.elu(x)

        for conv in self.convs[:-1]:
            x = conv(x, edge_index, edge_attr=edge_attr) # adding edge features here!
            x = self.elu(x)
      
        x = self.convs[-1](x, edge_index, edge_attr=edge_attr)
        x = self.classifier(x)
        return x