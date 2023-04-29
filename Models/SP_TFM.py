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
    def __init__(self, seq_len, nfeat, nhid, nheads, ntfm, dropout):
        """Dense version of GAT."""
        super(SP_TFM, self).__init__()
        self.linear = nn.Linear(nfeat, nhid * nheads)
        self.encoder = nn.TransformerEncoderLayer(d_model=nhid*nheads, nhead=nheads, dropout=dropout, dim_feedforward=nhid*nheads, batch_first=True)
        self.transformer_enc = nn.TransformerEncoder(self.encoder, num_layers=ntfm)
        self.pos_encoding = nn.Parameter(torch.randn(1, seq_len, nhid*nheads))
        self.out = nn.Linear(nhid * nheads, 1)
    def forward(self, x):
        x = x[:, :, 2:]
        x = self.linear(x)
        x += self.pos_encoding
        x = self.transformer_enc(x)
        x = self.out(x)
        return x
    
class SP_TFM_REL(nn.Module):
    '''
    Pure Global aggregation using transformers
    Deterministic Positional Encoding 
    '''
    def __init__(self, nfeat, dilation, nhid, nheads, ntfm, dropout):
        """Dense version of GAT."""
        super(SP_TFM_REL, self).__init__()
        self.linear = nn.Linear(nfeat, nhid * nheads)
        self.pos_linear = nn.Linear(2, nhid*nheads)

        # self.pos_encoding = PositionalEncodingSuperPixel(nhid*nheads)
        self.transformer_enc = PosTransformer(nhid * nheads, dilation, ntfm, nheads, nhid, nhid*nheads, dropout)
        # self.encoder = nn.TransformerEncoderLayer(d_model=nhid*nheads, nhead=nheads, dropout=dropout, dim_feedforward=nhid*nheads, batch_first=True)
        # self.transformer_enc = nn.TransformerEncoder(self.encoder, num_layers=ntfm)

        self.out = nn.Linear(nhid * nheads, 1)
    def forward(self, x, adj, distances):
        pos = x[:, :, :2]
        x = x[:, :, 2:]
        x = self.linear(x)
        pos = self.pos_linear(pos)
        # pos = self.pos_encoding(pos)
        x += pos
        x = self.transformer_enc(x, None, adj, distances)

        x = self.out(x)
        return x


class SP_TFM_FFT(nn.Module):
    '''
    Pure Global aggregation using transformers
    - Deterministic Positional Encoding 
    - Use Fourier descriptors as shape

    '''
    def __init__(self, nfeat, nhid, nheads, ntfm, dropout):
        """Dense version of GAT."""
        super(SP_TFM_FFT, self).__init__()
        self.linear = nn.Linear(nfeat-2, nhid * nheads)
        self.encoder = nn.TransformerEncoderLayer(d_model=nhid*nheads, nhead=nheads, dropout=dropout, dim_feedforward=nhid*nheads, batch_first=True)
        self.transformer_enc = nn.TransformerEncoder(self.encoder, num_layers=ntfm)
   

        self.pos_encoding = PositionalEncodingSuperPixel(nhid*nheads)
        self.out = nn.Linear(nhid * nheads, 1)
    def forward(self, x):
        centroids = x[:, :, :2]
        x = x[:, :, 2:]
        x = self.linear(x)

        x += self.pos_encoding(centroids)
        x = self.transformer_enc(x)
        x = self.out(x)
        return x


class SP_TFM_Contour(nn.Module):
    '''
    Pure Global aggregation using transformers
    - Deterministic Positional Encoding 
    - Using contours as shapes
    
    '''
    def __init__(self, nfeat, nhid, nheads, ntfm, dropout):
        """Dense version of GAT."""
        super(SP_TFM_Contour, self).__init__()
        shape_dim = 64
        self.linear = nn.Linear(nfeat-2, nhid * nheads-shape_dim)
        self.encoder = nn.TransformerEncoderLayer(d_model=nhid*nheads, nhead=nheads, dropout=dropout, dim_feedforward=nhid*nheads, batch_first=True)
        self.transformer_enc = nn.TransformerEncoder(self.encoder, num_layers=ntfm)

        # Shape embeddings
        
        self.shape_linear = nn.Linear(2, shape_dim)
        self.shape_token = nn.Parameter(torch.randn(1, 1, shape_dim))
        self.shape_encoder = nn.TransformerEncoderLayer(d_model = shape_dim, nhead=1, dropout=dropout, dim_feedforward=shape_dim, batch_first=True)
        self.shape_tfm = nn.TransformerEncoder(self.shape_encoder, num_layers=1)

        self.pos_encoding = PositionalEncodingSuperPixel(nhid*nheads)
        self.out = nn.Linear(nhid * nheads, 1)
    def forward(self, x):
        centroids = x[:, :, :2]
        shape = x[:, :, 2:2+(RESAMPLE_POINTS*2)]
        shape = shape.reshape(-1, RESAMPLE_POINTS, 2)
        x = x[:, :, 2+(RESAMPLE_POINTS*2):]
        x = self.linear(x)

        b, n, _ = shape.size()
        shape = self.shape_linear(shape)
        shape_tokens = repeat(self.shape_token, '1 1 d -> b 1 d', b=b)
        shape = torch.cat((shape_tokens, shape), dim=1)
        shape = self.shape_tfm(shape)
        shape = shape[:, 0:1, :]
        shape = shape.reshape(x.size(0), -1, shape.size(-1))

        pos_encoding = self.pos_encoding(centroids)

        x = torch.cat((x, shape), dim=2)
        x = self.transformer_enc(x+pos_encoding)
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
    def __init__(self, nfeat, nhid, block_depth, dropout, nheads, ntfm, norm='ln'):
        """Dense version of GAT."""
        super(SP_TFM_TFM, self).__init__()
        self.linear = nn.Linear(nfeat-2, nhid * nheads)
        self.transformers = nn.ModuleList([GraphConvTransformer(nhid*nheads, block_depth, nheads, nhid, nheads*nhid, norm=norm, dropout=dropout) for _ in range(ntfm)])
        self.pos_encoding = PositionalEncodingSuperPixel(nhid*nheads)
        self.out = nn.Linear(nhid * nheads, 1)
    def forward(self, input):
        x = input[0]
        adj = input[1]
        centroids = x[:, :, :2]
        x = x[:, :, 2:]
        x = self.linear(x)
        x += self.pos_encoding(centroids)
        # x = torch.cat((centroids, x), dim=2)
        for layer in self.transformers:
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


class SP_MNIST_TFM(nn.Module):
    '''
    Pure Global aggregation using transformers
    Deterministic Positional Encoding 
    '''
    def __init__(self, nfeat, max_len, nhid, nheads, ntfm, dropout):
        """Dense version of GAT."""
        super(SP_MNIST_TFM, self).__init__()
        self.linear = nn.Linear(nfeat-2, nhid * nheads)
        self.pos_linear = nn.Linear(2, nhid)
        self.cls_token = nn.Parameter(torch.randn(1, 1, nhid*nheads))
        self.transformer_enc = PosTransformer(nhid * nheads, max_len, ntfm, nheads, nhid, nhid*nheads, dropout)

        self.out = nn.Linear(nhid * nheads, 10)
    def forward(self, x):
        centroids = x[:, :, :2]
        x = x[:, :, 2:]
        x = self.linear(x)
        pos = self.pos_linear(centroids)
        pos = F.pad(pos, (0, 0, 1, 0), 'constant', 0)
       
        cls_tokens = self.cls_token.repeat(x.size(0), 1, 1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.transformer_enc(x, pos)
        # x = self.transformer_dec(x, x)
        x = x[:, 0] # B, D
        x = self.out(x)
        return x