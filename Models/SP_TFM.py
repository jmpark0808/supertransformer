import torch.nn as nn
from Blocks.GraphBlocks import *
from Wrappers.PositionalEncoding import PositionalEncodingSuperPixel
from Blocks.TransformerBlocks import *


class SP_TFM_TFM(nn.Module):
    '''
    Graph Convolutions using Transformers + Global aggregation using transformers
    Deterministic Positional Encoding 
    '''
    def __init__(self, nfeat, nhid, block_depth, dropout, nheads, ntfm, num_regions, norm='ln'):
        """Dense version of GAT."""
        super(SP_TFM_TFM, self).__init__()
        self.linear = nn.Linear(nfeat-2, nhid * nheads-2)
        self.transformers = nn.ModuleList([GraphConvTransformer(nhid*nheads, block_depth, nheads, nhid, nheads*nhid, num_regions, norm=norm, dropout=dropout) for _ in range(ntfm)])
        # self.pos_encoding = PositionalEncodingSuperPixel(nhid*nheads)
        self.out = nn.Linear(nhid * nheads, 1)
    def forward(self, x, adj):
        centroids = x[:, :, :2]
        x = x[:, :, 2:]
        x = self.linear(x)
        x = torch.cat((centroids, x), dim=2)
        for layer in self.transformers:
            # x += self.pos_encoding(x)
            x = layer(x, adj)
        x = self.out(x)
        return x

class SP_RTFM_TFM(nn.Module):
    '''
    Graph Convolutions using Transformers + Global aggregation using transformers
    No positional encoding
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

class SP_ETFM_TFM(nn.Module):
    '''
    Graph Convolutions using Transformers + Global aggregation using transformers
    Deterministic Positional Encoding 
    '''
    def __init__(self, nfeat, nhid, block_depth, dropout, nheads, num_regions):
        """Dense version of GAT."""
        super(SP_ETFM_TFM, self).__init__()
        self.linear = nn.Linear(nfeat, nhid * nheads)
        # self.conv_transformers = nn.ModuleList([GraphConvETransformer(nhid*nheads, block_depth, nheads, nhid, nheads*nhid, num_regions, i, norm=norm, dropout=dropout) for i in range(ntfm)])
        dim = nhid*nheads
        heads = nheads
        dim_head = nhid
        mlp_dim = nheads*nhid
        self.conv_transformers = nn.ModuleList([])
        for l in range(block_depth):
            self.conv_transformers.append(nn.ModuleList([
                PreBatchNorm(dim, GraphEAttention(dim, num_regions, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreBatchNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
        self.global_transformer = nn.ModuleList([])
        for g in range(2):
            self.global_transformer.append(nn.ModuleList([
                PreBatchNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreBatchNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
        # self.pos_encoding = PositionalEncodingSuperPixel(nhid*nheads)
        self.out = nn.Linear(nhid * nheads, 1)
    def forward(self, x, adj):
        centroids = x[:, :, :2]
        shape_x = x[:, :, 11:11+NUM_CHUNK*2, None]
        shape_y = x[:, :, 11+NUM_CHUNK*2:, None]
        shape = torch.cat((shape_x, shape_y), dim=3)
        x = x[:, :, 2:11]
        x = self.linear(x)
        for attn, ff in self.conv_transformers:
            x = attn(x, adj=adj, dilation=1, cen=centroids, shape=shape) + x
            x = ff(x) + x
        for attn, ff in self.global_transformer:
            x = attn(x) + x
            x = ff(x) + x
        # for layer in self.conv_transformers:
        #     # x += self.pos_encoding(x)
        #     x = layer(x, adj)
        # for layer in self.conv_transformers:
        #     # x += self.pos_encoding(x)
        #     x = layer(x, adj)
        x = self.out(x)
        return x
