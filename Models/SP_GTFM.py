import torch.nn as nn
from Blocks.GraphBlocks import *
from Wrappers.PositionalEncoding import PositionalEncodingSuperPixel
from Blocks.TransformerBlocks import *
import matplotlib.pyplot as plt
from dataset.constants import *


class SP_GTFM(nn.Module):
    '''
    Pure Global aggregation using transformers
    Deterministic Positional Encoding 
    '''
    def __init__(self, nfeat, nhid, block_depth, dropout, nheads, ntfm, num_regions, norm='ln'):
        """Dense version of GAT."""
        super(SP_GTFM, self).__init__()
        self.linear = nn.Linear(nfeat-2, nhid * nheads)
        self.pos = nn.Linear(POS_EMBEDDING, nhid*nheads)
        self.edge = nn.Linear(4, nhid*nheads)

        
        self.layers = nn.ModuleList([GraphTransformer(nhid*nheads, block_depth, nheads, nhid, nheads*nhid, num_regions, norm=norm, dropout=dropout) for _ in range(ntfm)])

        self.out = nn.Linear(nhid * nheads, 1)
    def forward(self, node_features, pos_enc, edge_features):
        x = x[:, :, 2:]
        x = self.linear(x)
        lap_pos_enc = self.pos(pos_enc)
        x += lap_pos_enc
        e = self.edge(edge_features)
        for conv in self.layers:
            x, e = conv(x, e)

        x = self.out(x)
        return x
