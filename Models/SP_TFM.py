import torch.nn as nn
from Blocks.GraphBlocks import *
from Wrappers.PositionalEncoding import PositionalEncodingSuperPixel
from Blocks.TransformerBlocks import *
import matplotlib.pyplot as plt
from dataset.constants import *


class SP_TFM(nn.Module):
    '''
    Pure Global aggregation using transformers
    Deterministic Positional Encoding 
    '''
    def __init__(self, nfeat, nhid, nheads, ntfm):
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
    def forward(self, x):
        centroids = x[:, :, :2]
        x = x[:, :, 2:]
        x = self.linear(x)
  
        x += self.pos_encoding(centroids)
        x = self.transformer_enc(x)
        # x = self.transformer_dec(x, x)
        x = self.out(x)
        return x

class SP_TFM_NP(nn.Module):
    '''
    Pure Global aggregation using transformers
    Deterministic Positional Encoding 
    '''
    def __init__(self, nfeat, nhid, block_depth, dropout, nheads, ntfm, num_regions, norm='ln'):
        """Dense version of GAT."""
        super(SP_TFM_NP, self).__init__()
        self.linear = nn.Linear(nfeat-2, nhid * nheads)
        self.encoder = nn.TransformerEncoderLayer(d_model=nhid*nheads, nhead=nheads, dim_feedforward=nhid*nheads, batch_first=True)
        self.transformer_enc = nn.TransformerEncoder(self.encoder, num_layers=ntfm)
        self.out = nn.Linear(nhid * nheads, 1)
    def forward(self, x):
        centroids = x[:, :, :2]
        x = x[:, :, 2:]
        x = self.linear(x)#+self.pos(centroids)
        x = self.transformer_enc(x)
        # x = self.transformer_dec(x, x)
        x = self.out(x)
        return x

class SP_TFM_AP(nn.Module):
    '''
    Pure Global aggregation using transformers
    Deterministic Positional Encoding 
    '''
    def __init__(self, nfeat, nhid, block_depth, dropout, nheads, ntfm, num_regions, norm='ln'):
        """Dense version of GAT."""
        super(SP_TFM_AP, self).__init__()
        self.linear = nn.Linear(nfeat, nhid * nheads)
        self.encoder = nn.TransformerEncoderLayer(d_model=nhid*nheads, nhead=nheads, dim_feedforward=nhid*nheads, batch_first=True)
        self.transformer_enc = nn.TransformerEncoder(self.encoder, num_layers=ntfm)
        self.out = nn.Linear(nhid * nheads, 1)
    def forward(self, x):
        x = self.linear(x)
        x = self.transformer_enc(x)
        x = self.out(x)
        return x

class SP_TFM_PE(nn.Module):
    '''
    Pure Global aggregation using transformers
    Deterministic Positional Encoding 
    '''
    def __init__(self, nfeat, nhid, block_depth, dropout, nheads, ntfm, num_regions, norm='ln'):
        """Dense version of GAT."""
        super(SP_TFM_PE, self).__init__()
        self.linear = nn.Linear(BINS*3, nhid * nheads)
        self.pos = nn.Linear(NUM_CHUNK*2, nhid * nheads)
        self.encoder = nn.TransformerEncoderLayer(d_model=nhid*nheads, nhead=nheads, dim_feedforward=nhid*nheads, batch_first=True)
        self.transformer_enc = nn.TransformerEncoder(self.encoder, num_layers=ntfm)
        self.pos_encoding = PositionalEncodingSuperPixel(nhid*nheads)
        self.out = nn.Linear(nhid * nheads, 1)
    def forward(self, x):
        centroids = x[:, :, :2]
        shape = x[:, :, 2+BINS*3:]/300.
        x_inp = x[:, :, 2:2+BINS*3]
        pos = self.pos(shape)
        x = self.linear(x_inp)#+self.pos(centroids)
        x += pos
        x = self.transformer_enc(x)
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
        self.encoder = nn.TransformerEncoderLayer(d_model=nhid*nheads, nhead=nheads, dim_feedforward=nhid*nheads, dropout=dropout, batch_first=True)
        self.transformer_enc = nn.TransformerEncoder(self.encoder, num_layers=block_depth)
        self.out = nn.Linear(nhid * nheads, 1)
    def forward(self, x, adj):
        x = self.linear(x)
        x = self.transformer_enc(x)
        x = self.out(x)
        return x


