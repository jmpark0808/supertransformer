import torch.nn as nn
from Blocks.GraphBlocks import *
from Wrappers.PositionalEncoding import PositionalEncodingSuperPixel
from Blocks.TransformerBlocks import *

class SP_TFM(nn.Module):
    '''
    Pure Global aggregation using transformers
    Deterministic Positional Encoding 
    '''
    def __init__(self, nfeat, nhid, block_depth, dropout, nheads, ntfm, num_regions, norm='ln'):
        """Dense version of GAT."""
        super(SP_TFM, self).__init__()
        self.linear = nn.Linear(nfeat-2, nhid * nheads)
        # self.pos = nn.Linear(2, nhid*nheads)
        self.encoder = nn.TransformerEncoderLayer(d_model=nhid*nheads, nhead=nheads, dim_feedforward=nhid*nheads, batch_first=True)
        self.transformer_enc = nn.TransformerEncoder(self.encoder, num_layers=ntfm)
        # self.transformers_enc = Transformer(nhid*nheads, ntfm, nheads, nhid, nhid*nheads, dropout)
        # self.transformers = nn.ModuleList([GraphConvTransformer(nhid*nheads, block_depth, nheads, nhid, nheads*nhid, num_regions, norm=norm, dropout=dropout) for _ in range(ntfm)])
        # self.decoder = nn.TransformerDecoderLayer(d_model=nhid*nheads, nhead=nheads, dim_feedforward=nhid*nheads, batch_first=True)
        # self.transformer_dec = nn.TransformerDecoder(self.decoder, num_layers=ntfm)
        self.pos_encoding = PositionalEncodingSuperPixel(nhid*nheads)
        self.out = nn.Linear(nhid * nheads, 1)
    def forward(self, x, adj):
        centroids = x[:, :, :2]
        x = x[:, :, 2:]
        x = self.linear(x)#+self.pos(centroids)

        x += self.pos_encoding(centroids)
        x = self.transformer_enc(x)
        # x = self.transformer_dec(x, x)
        x = self.out(x)
        return x


class SP_TFM_TFM(nn.Module):
    '''
    Graph Convolutions using Transformers + Global aggregation using transformers
    Deterministic Positional Encoding 
    '''
    def __init__(self, nfeat, nhid, block_depth, dropout, nheads, ntfm, num_regions, norm='ln'):
        """Dense version of GAT."""
        super(SP_TFM_TFM, self).__init__()
        self.linear = nn.Linear(nfeat, nhid * nheads)
        self.transformers = nn.ModuleList([GraphConvTransformer(nhid*nheads, block_depth, nheads, nhid, nheads*nhid, num_regions, norm=norm, dropout=dropout) for _ in range(ntfm)])
        self.pos_encoding = PositionalEncodingSuperPixel(nhid*nheads)
        self.out = nn.Linear(nhid * nheads, 1)
    def forward(self, x, adj):
        # centroids = x[:, :, :2]
        # x = x[:, :, 2:]
        x = self.linear(x)
        # x = torch.cat((centroids, x), dim=2)
        for layer in self.transformers:
            x += self.pos_encoding(x)
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
        self.heads = heads
        self.dilation = 1
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
        self.xy_to_pos = nn.Linear(NUM_CHUNK*2*2, dim_head, bias=False)
        self.eye = torch.eye(num_regions, device='cuda')
    def forward(self, x, adj):
        centroids = x[:, :, :2]
        shape_x = x[:, :, 11:11+NUM_CHUNK*2, None]
        shape_y = x[:, :, 11+NUM_CHUNK*2:, None]
        shape = torch.cat((shape_x, shape_y), dim=3)
        x = x[:, :, 2:11]
        x = self.linear(x)

        cen = centroids.unsqueeze(2).repeat(1, 1, NUM_CHUNK*2, 1)
        shape = shape + cen # absolute coordinates
        

        adj = torch.matrix_power(adj, self.dilation).bool().int()-torch.matrix_power(adj, self.dilation-1).bool().int()+self.eye
        adj_dots = adj.unsqueeze(1).bool() # B x 1 x R x R
        adj_dots = adj_dots.repeat(1, self.heads, 1, 1)
        adj_dists = adj.unsqueeze(3).unsqueeze(3).bool()
        adj_dists = adj_dists.repeat(1, 1, 1, NUM_CHUNK*2, 2)

        distances = shape[:,  None,:,  :, :]-cen[:, :, None, :, :] # B, 625, 625, 72, 2
        distances = torch.where(adj_dists > 0, distances, torch.zeros_like(distances)) # B, 625, 625, 72, 2
        distances = distances.reshape(distances.size(0), distances.size(1), distances.size(2), -1) # B, 625, 625, 144
        adj_dists = self.xy_to_pos(distances)

        for attn, ff in self.conv_transformers:
            x = attn(x, adj_dots=adj_dots, adj_dists=adj_dists) + x
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
