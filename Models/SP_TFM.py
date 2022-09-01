import torch.nn as nn
from Blocks.GraphBlocks import *
from Wrappers.PositionalEncoding import PositionalEncodingSuperPixel
from Blocks.TransformerBlocks import GraphDilatedConvTransformer, GraphConvTransformer


class SP_TFM_TFM(nn.Module):
    '''
    Graph Convolutions using Transformers + Global aggregation using transformers
    Deterministic Positional Encoding 
    '''
    def __init__(self, nfeat, nhid, block_depth, dropout, nheads, ntfm, norm='ln'):
        """Dense version of GAT."""
        super(SP_TFM_TFM, self).__init__()
        self.linear = nn.Linear(nfeat, nhid * nheads)
        self.transformers = nn.ModuleList([GraphConvTransformer(nhid*nheads, block_depth, nheads, nhid, nheads*nhid, norm=norm, dropout=dropout) for _ in range(ntfm)])
        self.pos_encoding = PositionalEncodingSuperPixel(nhid*nheads)
        self.out = nn.Linear(nhid * nheads, 1)
    def forward(self, x, adj):
        x = self.linear(x)
        for layer in self.transformers:
            x += self.pos_encoding(x)
            x = layer(x, adj)
        x = self.out(x)
        return x

class SP_RTFM_TFM(nn.Module):
    '''
    Graph Convolutions using Transformers + Global aggregation using transformers
    Deterministic Positional Encoding 
    '''
    def __init__(self, nfeat, nhid, block_depth, dropout, nheads, ntfm, num_regions, norm='ln'):
        """Dense version of GAT."""
        super(SP_RTFM_TFM, self).__init__()
        self.linear = nn.Linear(nfeat, nhid * nheads)
        self.transformers = nn.ModuleList([GraphConvTransformer(nhid*nheads, block_depth, nheads, nhid, nheads*nhid, num_regions, norm=norm, dropout=dropout) for _ in range(ntfm)])
        self.out = nn.Linear(nhid * nheads, 1)
    def forward(self, x, adj):
        x = self.linear(x)
        for layer in self.transformers:
            x = layer(x, adj)
        x = self.out(x)
        return x


class SP_TFM_DIL(nn.Module):
    '''
    Graph Convolutions using Transformers + Global aggregation using dilations
    Deterministic Positional Encoding 
    '''
    def __init__(self, nfeat, nhid, block_depth, dropout, nheads, ntfm, num_regions, norm='ln'):
        """Dense version of GAT."""
        super(SP_TFM_DIL, self).__init__()
        self.linear = nn.Linear(nfeat, nhid * nheads)
        self.transformers = nn.ModuleList([GraphDilatedConvTransformer(nhid*nheads, block_depth, nheads, nhid, nheads*nhid, num_regions, norm=norm, dropout=dropout) for _ in range(ntfm)])
        self.pos_encoding = PositionalEncodingSuperPixel(nhid*nheads)
        self.out = nn.Linear(nhid * nheads, 1)
    def forward(self, x, adj):
        x = self.linear(x)
        for layer in self.transformers:
            x += self.pos_encoding(x)
            x = layer(x, adj)
        x = self.out(x)
        return x


