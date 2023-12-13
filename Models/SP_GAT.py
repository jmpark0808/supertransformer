import torch.nn as nn
from Blocks.GraphBlocks import *
from Wrappers.PositionalEncoding import PositionalEncodingSuperPixel
from Blocks.TransformerBlocks import *
from dataset.constants import *


class SP_GAT(nn.Module):
    '''
    Pure Global aggregation using transformers
    Deterministic Positional Encoding 
    '''
    def __init__(self, nfeat, nhid, dropout, nheads, ntfm, alpha):
        """Dense version of GAT."""
        super(SP_GAT, self).__init__()
        self.linear = nn.Linear(nfeat-2, nhid * nheads)
        self.gat = [[GraphAttentionLayer(nhid*nheads, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)] for a in range(ntfm)]
        for i, block in enumerate(self.gat):
            for j, head in enumerate(block):
                self.add_module('attention_{}_{}'.format(i, j), head)
        self.pos_encoding = PositionalEncodingSuperPixel(nhid*nheads)
        self.out = GraphAttentionLayer(nhid*nheads, 1, dropout=dropout, alpha=alpha, concat=False)
        
    def forward(self, input):
        x = input[0]
        adj = input[1]
        centroids = x[:, :, :2]
        x = x[:, :, 2:]
        x = self.linear(x)#+self.pos(centroids)
        x += self.pos_encoding(centroids)
        for block in self.gat:
            x = torch.cat([head(x, adj) for head in block], dim=2)
                
        x = self.out(x, torch.ones([x.size(0), x.size(1), x.size(1)], device=x.device))
        return x
    
class SP_GATv2(nn.Module):
    '''
    Graph Attention Network v2
    '''
    def __init__(self, nfeat, nhid, dropout, nheads, ntfm):
        """Dense version of GAT."""
        super(SP_GATv2, self).__init__()
        self.linear = nn.Linear(nfeat, nhid)
        self.gat = nn.ModuleList([nn.ModuleList([GATv2(nhid, nhid, dropout, nheads, 0.2, False), nn.LayerNorm(nhid)]) for _ in range(ntfm)])
        self.pos_encoding = nn.Linear(2, nhid)
        self.out = nn.Linear(nhid, 1)
        self.relu = nn.ReLU()
        
    def forward(self, x, adj=None):
        centroids = x[:, :, :2]
        x = x[:, :, 2:]
        x = self.linear(x)
        x = self.relu(x)

        x = x + self.pos_encoding(centroids)
        for block, ln in self.gat:
            x = ln(x)
            x = block(x, adj)
            x = F.relu(x)
                
        x = self.out(x)
        return x
    
class SP_GATv3(nn.Module):
    '''
    Graph Attention Network v2
    '''
    def __init__(self, nfeat, nhid, dropout, nheads, ntfm):
        """Dense version of GAT."""
        super(SP_GATv3, self).__init__()
        self.linear = nn.Linear(nfeat, nhid)
        self.gat = nn.ModuleList([nn.ModuleList([GATv3(nhid, nhid, dropout, nheads, 0.2, False), nn.LayerNorm(nhid)]) for _ in range(ntfm)])
        self.pos_encoding = nn.Linear(2, nhid)
        self.out = nn.Linear(nhid, 1)
        self.relu = nn.ReLU()
        
    def forward(self, x, adj=None):
        centroids = x[:, :, :2]
        x = x[:, :, 2:]
        x = self.linear(x)
        x = self.relu(x)

        x = x + self.pos_encoding(centroids)
        for block, ln in self.gat:
            x = ln(x)
            x = block(x, adj)
            x = F.relu(x)
                
        x = self.out(x)
        return x
