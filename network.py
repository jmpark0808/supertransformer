# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import time
import numpy as np
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

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

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
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

# class Transformer(nn.Module):
#     def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
#         super().__init__()
#         self.layers = nn.ModuleList([])
#         for _ in range(depth):
#             self.layers.append(nn.ModuleList([
#                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
#                FeedForward(dim, mlp_dim, dropout = dropout)
#             ]))
#     def forward(self, x):
#         for attn, ff in self.layers:
#             x = attn(x) + x
#             x = ff(x) + x
#         return x




class PositionalEncodingSuperPixel(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncodingSuperPixel, self).__init__()
        channels = int(np.ceil(channels / 4) * 2)
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, tensor):
        """
        :param tensor: A 3d tensor of size (batch_size, seq_len, features)
        :return: Positional Encoding Matrix of size (batch_size, seq_len, features)
        """
        if len(tensor.shape) != 3:
            raise RuntimeError("The input tensor has to be 3d!")
        batch_size, seq, feat = tensor.shape
        pos_x = tensor[:, :, 0].type(self.inv_freq.type())
        pos_y = tensor[:, :, 1].type(self.inv_freq.type())

        sin_inp_x = torch.einsum("bi,j->bij", pos_x, self.inv_freq) # batch, seq, feat/4
        sin_inp_y = torch.einsum("bi,j->bij", pos_y, self.inv_freq) # batch, seq, feat/4
        emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1) # batch, seq, feat/2
        emb_y = torch.cat((sin_inp_y.sin(), sin_inp_y.cos()), dim=-1) # batch, seq, feat/2
        emb = torch.zeros((batch_size, seq, self.channels * 2), device=tensor.device).type(
            tensor.type()
        )
        emb[:, :, : self.channels] = emb_x
        emb[:, :, self.channels : 2 * self.channels] = emb_y

        return emb

class SuperT(nn.Module):
    def __init__(self, *, feature_dim, seq_len, dim, depth, heads, mlp_dim, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
 
        # self.to_SP_embedding = nn.Linear(feature_dim, dim)
        

        # self.pos_encoding = PositionalEncodingSuperPixel(dim)
        # self.pos_encoding = nn.Parameter(torch.randn(1, seq_len, dim))

        # self.dropout = nn.Dropout(emb_dropout)

        self.conv_transformer = Transformer(feature_dim, 1, heads, dim_head, mlp_dim, dropout)
        self.global_transformer = Transformer(feature_dim, 1, heads, dim_head, mlp_dim, dropout)

        self.mlp_head = nn.Linear(dim, 1)

    def forward(self, x, nb_indices): # img = (batch, seq_len, feature_dim)
        # pos_encoding = self.pos_encoding(img)  # (batch, seq_len, dim))
        #x = self.to_SP_embedding(img[:, :, 3:6]) # (batch, seq_len, dim)

 
        #x += self.pos_encoding
        #x = self.dropout(x)
        x = torch.unsqueeze(x, 2)
        x = x.expand(-1, -1, x.size(1), -1)
        nb_indices = torch.unsqueeze(nb_indices, 3)
        x = torch.gather(x, 1, nb_indices) # (batch, seq_len, 9, feature_dim)
        x = self.conv_transformer(x) # (batch, seq_len, 9, feature_dim)

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