import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


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

    def forward(self, h, adj):
        Wh = torch.matmul(h, self.W) # h.shape: (B, N, in_features), Wh.shape: (B, N, out_features)
        e = self._prepare_attentional_mechanism_input(Wh)

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=-1)
        attention = self.dropout(attention)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (B, N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (B, N, 1)
        # e.shape (B, N, N)
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
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



