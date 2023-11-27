# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import time
import numpy as np
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from Wrappers.PositionalEncoding import PositionalEncodingSuperPixel
from dataset.constants import *

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class PreBatchNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.BatchNorm1d(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        x = x.permute(0, 2, 1) # Batch, channels, seq_len
        x = self.norm(x)
        x = x.permute(0, 2, 1) # Batch, seq_len, channels
        return self.fn(x, **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0., out_dim=None):
        super().__init__()
        if out_dim is None:
            self.net = nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, dim),
                nn.Dropout(dropout)
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, out_dim),
                nn.Dropout(dropout)
            )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2))  #* self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class PosAttention(nn.Module):
    def __init__(self, dim, dilation, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.dilation = dilation

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.distances_linear = nn.Linear(2, dim_head)
        # self.distances_1 = nn.Linear(dim_head*heads, 1)


        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, emb, adj, distances):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
        
        dots = torch.matmul(q, k.transpose(-1, -2))
        zero_vec = -1e9*torch.ones_like(dots)

        # adj = torch.matrix_power(adj, self.dilation).bool().int()
        adj = adj.unsqueeze(1).bool() # B x 1 x R x R
        adj = adj.repeat(1, dots.size(1), 1, 1)

        if emb is not None:
            Er_t = emb.transpose(1, 2).unsqueeze(1)
            QEr = torch.matmul(q, Er_t)
            Srel = self.skew(QEr)
            attention = torch.where(adj > 0, dots+Srel, zero_vec)
        # elif distances is not None:
        #     distances = torch.relu(self.distances_linear(distances))
        #     distances = self.distances_1(distances).reshape(distances.size(0), 1, distances.size(1), distances.size(2))
        #     # QEr = torch.einsum('abcd,aced->abce', q, distances)
        #     # attention = torch.where(adj > 0, dots+QEr, zero_vec)
        #     attention = torch.where(adj > 0, dots+distances, zero_vec)
        else:
            attention = torch.where(adj > 0, dots, zero_vec)

        attn = self.attend((attention)*self.scale)
        

        
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        
        return self.to_out(out)
    
    def skew(self, QEr):
        # QEr.shape = (batch_size, num_heads, seq_len, seq_len)
        padded = F.pad(QEr, (1, 0))
        # padded.shape = (batch_size, num_heads, seq_len, 1 + seq_len)
        batch_size, num_heads, num_rows, num_cols = padded.shape
        reshaped = padded.reshape(batch_size, num_heads, num_cols, num_rows)
        # reshaped.size = (batch_size, num_heads, 1 + seq_len, seq_len)
        Srel = reshaped[:, :, 1:, :]
        # Srel.shape = (batch_size, num_heads, seq_len, seq_len)
        return Srel


class GraphAttention(nn.Module):
    def __init__(self, dim,  heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
  

    def forward(self, x, adj):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale # seq_len x seq_len 
        zero_vec = 0*torch.ones_like(dots)

        # adj = torch.matrix_power(adj, dilation).bool().int()-torch.matrix_power(adj, dilation-1).bool().int()+self.eye
        adj = adj.unsqueeze(1).bool() # B x 1 x R x R
        adj = adj.repeat(1, dots.size(1), 1, 1)

        attention = torch.where(adj > 0, dots, zero_vec)
        
        attn = self.attend(attention)
        # all_out = []
        # for j in range(v.size(0)):
        #     for i in range(v.size(1)):
        #         Ats = torch_sparse.SparseTensor.from_torch_sparse_coo_tensor(attn[j, i].to_sparse())
        #         out_ = torch_sparse.matmul(Ats,v[j, i])
        #         # temp_attn = attn[j, i].to_sparse()
        #         # out_ = torch_sparse.spmm(temp_attn.indices(), temp_attn.values(), attn[j, i].size(0), attn[j, i].size(1), v[j, i])
        #         all_out.append(out_)
        # out = torch.stack(all_out, dim=0)
        # out = out.reshape(v.size(0), v.size(1), v.size(2), -1)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class GraphEAttention(nn.Module):
    def __init__(self, dim, num_regions, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
        

    def forward(self, x, adj_dots, adj_dists):
        # cen = B, 625, 2
        # shape = B, 625, 72, 2
        
        

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) #* self.scale # seq_len x seq_len 
        zero_vec = -9e15*torch.ones_like(dots)

        attention1 = torch.where(adj_dots > 0, dots, zero_vec)
        

        attention2 = torch.matmul(q.permute(0, 2, 1, 3), adj_dists.permute(0, 1, 3, 2)).permute(0, 2, 1, 3) # B, H, Nq, Nk
        
        attn = self.attend(attention1+attention2)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class PosTransformer(nn.Module):
    def __init__(self, dim, dilation, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, PosAttention(dim, dilation, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x, emb, adj, distances):
        for idx, (attn, ff) in enumerate(self.layers):
            if idx == 0:
                x = attn(x, emb=emb, adj=adj, distances=distances) + x
                x = ff(x) + x
            else:
                x = attn(x, emb=emb, adj=adj, distances=None) + x
                x = ff(x) + x
        return x

class GraphConvTransformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, norm=True, dropout = 0.):
        super().__init__()
        self.graph_conv_layers = nn.ModuleList([])
        if norm=='ln':
            for _ in range(depth):
                self.graph_conv_layers.append(nn.ModuleList([
                    PreNorm(dim, GraphAttention(dim,  heads = heads, dim_head = dim_head, dropout = dropout)),
                    PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
                ]))
        elif norm=='bn':
            for _ in range(depth):
                self.graph_conv_layers.append(nn.ModuleList([
                    PreBatchNorm(dim, GraphAttention(dim,  heads = heads, dim_head = dim_head, dropout = dropout)),
                    PreBatchNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
                ]))
        else:
            for _ in range(depth):
                self.graph_conv_layers.append(nn.ModuleList([
                    GraphAttention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                    FeedForward(dim, mlp_dim, dropout = dropout)
                ]))
    def forward(self, x, adj):
        for attn, ff in self.graph_conv_layers:
            x = attn(x, adj=adj) + x
            x = ff(x) + x
        return x

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, dropout):
        super().__init__()
        inner_dim = out_dim *  num_heads
        self.heads = num_heads
        self.scale = out_dim ** -0.5
        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(in_dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, in_dim),
            nn.Dropout(dropout)
        ) 

    def forward(self, x, e):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
        e = rearrange(e, 'b n k (h d) -> b h n k d', h = self.heads)

        dots = q.unsqueeze(3)*k.unsqueeze(2) * self.scale
        # dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale # seq_len x seq_len 

        dots = dots*e
        
        attn = self.attend(dots.sum(-1))

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out), dots


class GraphTransformerLayer(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout = 0.):
        super().__init__()
        inner_dim = heads*dim_head
        self.inner_dim = inner_dim
        self.attention = MultiHeadAttentionLayer(dim, dim_head, heads, dropout)
        self.O_h = nn.Linear(inner_dim, inner_dim)
        self.O_e = nn.Linear(inner_dim, inner_dim)

        self.bn1_h = nn.BatchNorm1d(inner_dim)
        self.bn1_e = nn.BatchNorm1d(inner_dim)

        self.bn2_h = nn.BatchNorm1d(inner_dim)
        self.bn2_e = nn.BatchNorm1d(inner_dim)

        self.ffn_h_layer = nn.Sequential(nn.Linear(inner_dim, inner_dim*2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(inner_dim*2, inner_dim))


        self.ffn_e_layer = nn.Sequential(nn.Linear(inner_dim, inner_dim*2), nn.ReLU(), nn.Dropout(dropout),nn.Linear(inner_dim*2, inner_dim))



     
    def forward(self, x, e):
        x_in = x
        e_in = e

        h_attn_out, e_attn_out = self.attention(x, e)
        e_attn_out = rearrange(e_attn_out, 'b h n k d -> b n k (h d)')

        h = self.O_h(h_attn_out)
        e = self.O_e(e_attn_out)
        h_size = h.size()
        e_size = e.size()

        h = x_in + h
        e = e_in + e

        h = h.view(-1, h_size[-1])
        e = e.view(-1, e_size[-1])
 
        h = self.bn1_h(h)
        e = self.bn1_e(e)

        h_in2 = h
        e_in2 = e

        h = self.ffn_h_layer(h)
        e = self.ffn_e_layer(e)

        h = h_in2 + h
        e = e_in2 + e

        h = self.bn2_h(h)
        e = self.bn2_e(e)

        h = h.view(h_size[0], h_size[1], -1)
        e = e.view(e_size[0], e_size[1], e_size[2], -1)
        return h, e


class GraphTransformer(nn.Module):
    def __init__(self, h_in, dim, heads, dim_head, depth, dropout = 0.):
        super().__init__()
        self.h_linear = nn.Sequential(nn.Linear(h_in, dim), nn.Dropout(dropout))
        # self.lap_pos_enc = nn.Linear(POS_EMBEDDING, dim)
        self.e_linear = nn.Linear(5, dim)
        self.layers = nn.ModuleList([GraphTransformerLayer(dim, heads, dim_head, dropout) for _ in range(depth)])
        self.out = nn.Linear(heads*dim_head, 1)
        self.pos_encoding = PositionalEncodingSuperPixel(dim_head*heads)

    def forward(self, h, e):
        h = self.h_linear(h[:, :, 2:])
        h = h + self.pos_encoding(h[:, :, :2])
        e = self.e_linear(e)
        for conv in self.layers:
            h, e = conv(h, e)
            
        return self.out(h)


class GraphConvETransformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, num_regions, block_ind, norm=True, dropout = 0.):
        super().__init__()
        self.graph_conv_layers = nn.ModuleList([])
        if norm=='ln':
            for l in range(depth):
                self.graph_conv_layers.append(nn.ModuleList([
                    PreNorm(dim, GraphEAttention(dim, num_regions, heads = heads, dim_head = dim_head, dropout = dropout, block_ind=block_ind, layer_ind=l)),
                    PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
                ]))
            self.graph_global_layers = nn.ModuleList([])
            for _ in range(1):
                self.graph_global_layers.append(nn.ModuleList([
                    PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                    PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
                ]))
        elif norm=='bn':
            for l in range(depth):
                self.graph_conv_layers.append(nn.ModuleList([
                    PreBatchNorm(dim, GraphEAttention(dim, num_regions, heads = heads, dim_head = dim_head, dropout = dropout, block_ind=block_ind, layer_ind=l)),
                    PreBatchNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
                ]))
            self.graph_global_layers = nn.ModuleList([])
            for _ in range(1):
                self.graph_global_layers.append(nn.ModuleList([
                    PreBatchNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                    PreBatchNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
                ]))
        else:
            for _ in range(depth):
                self.graph_conv_layers.append(nn.ModuleList([
                    GraphAttention(dim, num_regions, heads = heads, dim_head = dim_head, dropout = dropout),
                    FeedForward(dim, mlp_dim, dropout = dropout)
                ]))
            self.graph_global_layers = nn.ModuleList([])
            for _ in range(1):
                self.graph_global_layers.append(nn.ModuleList([
                    Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                    FeedForward(dim, mlp_dim, dropout = dropout)
                ]))
    def forward(self, x, adj):
        for attn, ff in self.graph_conv_layers:
            x = attn(x, adj=adj) + x
            x = ff(x) + x
        for attn, ff in self.graph_global_layers:
            x = attn(x) + x
            x = ff(x) + x
        return x



class GraphDilatedConvTransformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, num_regions, norm=True, dropout = 0.):
        super().__init__()
        self.graph_conv_layers = nn.ModuleList([])
        self.dilations = [1, 1, 1, 2, 2, 2, 4, 4, 4]
        assert len(self.dilations) == depth, "depth has to equal the length of dilations"
        if norm=='ln':
            for _ in range(depth):
                self.graph_conv_layers.append(nn.ModuleList([
                    PreNorm(dim, GraphAttention(dim, num_regions, heads = heads, dim_head = dim_head, dropout = dropout)),
                    PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
                ]))
        elif norm=='bn':
            for _ in range(depth):
                self.graph_conv_layers.append(nn.ModuleList([
                    PreBatchNorm(dim, GraphAttention(dim, num_regions, heads = heads, dim_head = dim_head, dropout = dropout)),
                    PreBatchNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
                ]))
        
        else:
            for _ in range(depth):
                self.graph_conv_layers.append(nn.ModuleList([
                    GraphAttention(dim, num_regions, heads = heads, dim_head = dim_head, dropout = dropout),
                    FeedForward(dim, mlp_dim, dropout = dropout)
                ]))
    def forward(self, x, adj):
        for ind, (attn, ff) in enumerate(self.graph_conv_layers):
            x = attn(x, adj=adj, dilation=self.dilations[ind]) + x
            x = ff(x) + x

        return x

class SuperT(nn.Module):
    def __init__(self, in_dim, feature_dim, depth, heads, mlp_dim, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
 
        # self.to_SP_embedding = nn.Linear(feature_dim, dim)
        

        # self.pos_encoding = PositionalEncodingSuperPixel(dim)
        # self.pos_encoding = nn.Parameter(torch.randn(1, seq_len, dim))

        # self.dropout = nn.Dropout(emb_dropout)

        self.lin_proj = nn.Linear(in_dim, feature_dim)

        self.global_transformer = Transformer(feature_dim, depth, heads, dim_head, mlp_dim, dropout)

        self.mlp_head = nn.Linear(feature_dim, 1)

    def forward(self, x): # img = (batch, seq_len, feature_dim)
        # pos_encoding = self.pos_encoding(img)  # (batch, seq_len, dim))
        #x = self.to_SP_embedding(img[:, :, 3:6]) # (batch, seq_len, dim)

 
        #x += self.pos_encoding
        #x = self.dropout(x)
        x = self.lin_proj(x)

        x = self.global_transformer(x) # (batch, seq_len, feature_dim)

        x = self.mlp_head(x)

        return x

class SuperTPos(nn.Module):
    def __init__(self, in_dim, seq_len, feature_dim, depth, heads, mlp_dim, width, height, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
 
        self.pos_encoding = PositionalEncodingDict(width, height, dim_head)

        self.lin_proj = nn.Linear(in_dim, feature_dim)

        self.global_transformer = PosTransformer(feature_dim, depth, heads, dim_head, mlp_dim, dropout)

        self.mlp_head = nn.Linear(feature_dim, 1)

    def forward(self, x): # img = (batch, seq_len, feature_dim)
        # pos_encoding = self.pos_encoding(img)  # (batch, seq_len, dim))
        #x = self.to_SP_embedding(img[:, :, 3:6]) # (batch, seq_len, dim)

 
        #x += self.pos_encoding
        #x = self.dropout(x)
        pos_x, pos_y = self.pos_encoding(x)
        x = self.lin_proj(x[:, :, 2:6])
        x = self.global_transformer(x, pos_x, pos_y) # (batch, seq_len, feature_dim)

        x = self.mlp_head(x)

        return x

class MLP(nn.Module):
    def __init__(self, seq_len, channels):
        super().__init__()
 
        self.linear1 = nn.Linear(seq_len*channels, 512)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(512, seq_len)
        


    def forward(self, img): # img = (batch, seq_len, feature_dim)
        dim = img.size()
        img = img.reshape(dim[0], -1)
        x = self.linear1(img)
        x = self.relu1(x)
        x = self.linear2(x)


        return x