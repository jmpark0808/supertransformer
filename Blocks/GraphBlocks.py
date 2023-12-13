import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time

class MaxPoolingAggregator(nn.Module):
    """
    Max-pooling layer for graph convolutional neural networks
    """
    def __init__(self, in_features, hidden_dim, out_features, seq_len, dropout=1., bias=False):
        super().__init__()
        self.mlp_layer = nn.Linear(in_features, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()
        self.bias = bias
        self.eye = torch.eye(seq_len, device='cuda')
        self.neigh_weights = nn.Parameter(torch.randn(hidden_dim, out_features))
        self.self_weights = nn.Parameter(torch.randn(in_features, out_features))
        if bias:
            self.bias_operation = nn.Parameter(torch.zeros(out_features))
        
    def forward(self, x, adj):
        neighbours_only = adj-self.eye
        neigh_h = self.mlp_layer(x)
        neigh_h = torch.einsum('bij,bjk->bijk', neighbours_only, neigh_h)
        neigh_h = neigh_h.max(dim=2)[0]

        from_neighs = torch.matmul(neigh_h, self.neigh_weights)
        from_self = torch.matmul(x, self.self_weights)

        output = from_self + from_neighs

        if self.bias:
            output = output + self.bias_operation

        return self.act(output)

class MaxPoolingCNN(nn.Module):
    """
    Max-pooling layer for graph convolutional neural networks
    """
    def __init__(self, resolution, in_features, hidden_dim, out_features, kernel, dilation, dropout=1., bias=False):
        super().__init__()
        self.mlp_layer = nn.Linear(in_features, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()
        self.bias = bias

        self.resolution = resolution
        self.dilation = dilation
        self.kernel = kernel
        self.padding = int(((resolution-1)*1-resolution+kernel+(kernel-1)*(dilation-1))/2)

        self.neigh_weights = nn.Parameter(torch.randn(hidden_dim, out_features))
        self.self_weights = nn.Parameter(torch.randn(in_features, out_features))
        if bias:
            self.bias_operation = nn.Parameter(torch.zeros(out_features))
        
    def forward(self, x):
        x = x.permute(0, 2, 3, 1) # batch, X, Y, in_features
        neigh_h = self.mlp_layer(x) # batch, X, Y, hidden_dim
        size = neigh_h.size()
        neigh_h = F.unfold(neigh_h, self.kernel, self.dilation, self.padding, 1) # batch, hidden_dim * kernel^2, XY
        neigh_h = neigh_h.reshape(size[0], size[3], self.kernel*self.kernel, size[1], size[2])
        neigh_h[:, :, int(self.kernel*self.kernel/2), :, :] = -1e10
        neigh_h = neigh_h.max(dim=2)[0] # batch, hidden_dim, X, Y
        neigh_h = neigh_h.permute(0, 2, 3, 1) # batch, X, Y, hidden_dim

        from_neighs = torch.matmul(neigh_h, self.neigh_weights) # batch, X, Y, out_features
        
        from_self = torch.matmul(x, self.self_weights) # batch, X, Y, out_features

        output = from_self + from_neighs

        if self.bias:
            output = output + self.bias_operation

        output = output.permute(0, 3, 1, 2)

        return self.act(output)

class Matmul(nn.Module):
    def forward(self, *args):
        return torch.matmul(*args)

class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout, concat=True, alpha=0.2):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.dropout = nn.Dropout(dropout)
        self.matmul = Matmul()

    def forward(self, h, adj):
        Wh = self.matmul(h, self.W) # h.shape: (B, N, in_features), Wh.shape: (B, N, out_features)
        e = self._prepare_attentional_mechanism_input(Wh)

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=-1)
        attention = self.dropout(attention)
        h_prime = self.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (B, N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (B, N, 1)
        # e.shape (B, N, N)
        Wh1 = self.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = self.matmul(Wh, self.a[self.out_features:, :])
        # broadcast add
        e = Wh1 + Wh2.permute(0, 2, 1)
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GraphConvolutionBlock(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, nfeat, nhid, dropout, nheads, seq_len):
        super(GraphConvolutionBlock, self).__init__()
        self.dropout = dropout
        self.attention = [GraphAttentionLayer(nfeat, nhid, dropout=dropout,  concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attention):
            self.add_module('attention_{}'.format(i), attention)
        self.maxpool = MaxPoolingAggregator(nhid*nheads, nhid*nheads, nhid*nheads, seq_len, dropout, bias=True)


    def forward(self, x, adj):
        x = torch.cat([att(x, adj) for att in self.attention], dim=2)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.maxpool(x, adj)
        return x

class GATv2(nn.Module):
    """
    GATv2 similar to GATv2conv from pytorch geometric
    """
    def __init__(self, in_channels, out_channels, dropout, nheads, negative_slope, concat):
        super(GATv2, self).__init__()
        self.lin = nn.Linear(in_channels, out_channels)
        self.lin_l = nn.Linear(out_channels, nheads)
        self.lin_r = nn.Linear(out_channels, nheads)
        self.concat = concat

        self.heads = nheads
        self.out_channels = out_channels
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.dropout = nn.Dropout(dropout)


    def forward(self, x, adj):
  
        H, C = self.heads, self.out_channels
        x = self.lin(x) #B, N, C
        x_l = self.lin_l(x) # B, N, H
        x_r = self.lin_r(x) # B, N, H
        

        x_l = x_l.permute(0, 2,  1).unsqueeze(-1) + x_r.permute(0, 2, 1).unsqueeze(2) # B, H, N, N
        
        # x = torch.matmul(x_l.permute(0, 2, 1, 3), x_r.permute(0, 2, 3, 1)) #x_l.unsqueeze(2) + x_r.unsqueeze(1) # B, H, N, N
        x_l = self.leaky_relu(x_l)
        # return x
        # alpha = (x.unsqueeze(2)*self.att).sum(2) # B, H, N, N
        
        alpha = torch.softmax(x_l, -1) 

        alpha = self.dropout(alpha)

    
        out = torch.matmul(alpha, x.unsqueeze(1).repeat(1, H, 1, 1)) # B, H, N, C
        
        if self.concat:
            out = out.permute(0, 2, 1, 3) # B, N, H, C
            out = out.reshape(out.size(0), out.size(1), -1)# B, N, H*C
        else:
            out = torch.mean(out, dim=1) # B, N, C

        return out
    

class GATv3(nn.Module):
    """
    GATv2 similar to GATv2conv from pytorch geometric
    """
    def __init__(self, in_channels, out_channels, dropout, nheads, negative_slope, concat):
        super().__init__()
        self.lin_l = nn.Linear(in_channels, nheads*out_channels)
        self.lin_r = nn.Linear(in_channels, nheads*out_channels)
        self.att = nn.Parameter(torch.empty(1, nheads, out_channels, 1, 1))
        self.concat = concat
        if concat:
            self.bias = nn.Parameter(torch.empty(1, 1, nheads*out_channels))
        else:
            self.bias = nn.Parameter(torch.empty(1, 1, out_channels))

        self.heads = nheads
        self.out_channels = out_channels
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.dropout = nn.Dropout(dropout)


    def forward(self, x, adj):
  
        H, C = self.heads, self.out_channels
        x_l = self.lin_l(x).view(x.size(0), -1, H, C) # B, N, H, C
        x_r = self.lin_r(x).view(x.size(0), -1, H, C) # B, N, H, C
        

        x = x_l.permute(0, 2, 3,  1).unsqueeze(-1) + x_r.permute(0, 2, 3, 1).unsqueeze(3) # B, H, C, N, N
        
        # x = torch.matmul(x_l.permute(0, 2, 1, 3), x_r.permute(0, 2, 3, 1)) #x_l.unsqueeze(2) + x_r.unsqueeze(1) # B, H, N, N
        x = self.leaky_relu(x)
        # return x
        alpha = (x*self.att).sum(2) # B, H, N, N
        
        alpha = torch.softmax(alpha, -1) 

        alpha = self.dropout(alpha)

        out = torch.matmul(alpha, x_r.permute(0, 2, 1, 3)) # B, H, N, C
        
        if self.concat:
            out = out.permute(0, 2, 1, 3) # B, N, H, C
            out = out.reshape(out.size(0), out.size(1), -1)# B, N, H*C
        else:
            out = torch.mean(out, dim=1) # B, N, C

        out = out + self.bias
        return out
    



# class GATv2(nn.Module):
#     """
#     GATv2 similar to GATv2conv from pytorch geometric
#     """
#     def __init__(self, in_channels, out_channels, dropout, nheads, concat):
#         super(GATv2, self).__init__()
#         self.lin_q = nn.Linear(in_channels, nheads*out_channels)
#         self.lin_k = nn.Linear(in_channels, nheads*out_channels)
#         self.lin_v = nn.Linear(in_channels, nheads*out_channels)
#         self.concat = concat


#         self.heads = nheads
#         self.out_channels = out_channels

#         self.dropout = nn.Dropout(dropout)


#     def forward(self, x, adj):
#         H, C = self.heads, self.out_channels
#         x_q = self.lin_q(x).view(x.size(0), -1, H, C) # B, N, H, C
#         x_k = self.lin_k(x).view(x.size(0), -1, H, C) # B, N, H, C
#         x_v = self.lin_v(x).view(x.size(0), -1, H, C) # B, N, H, C

#         x = torch.matmul(x_q.permute(0, 2, 1, 3), x_k.permute(0, 2, 3, 1)) # B, H, N, N
#         alpha = torch.softmax(x, -1) 
#         alpha = self.dropout(alpha)
        
#         x = torch.matmul(alpha, x_v.permute(0, 2, 1, 3)) # B, H, N, C
#         if self.concat:
#             x = x.permute(0, 2, 1, 3) # B, N, H, C
#             x = x.reshape(x.size(0), x.size(1), -1)# B, N, H*C
#         else:
#             x = torch.mean(x, dim=1) # B, N, C

#         return x


