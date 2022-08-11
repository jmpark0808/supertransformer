import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from net.transformer import GraphConvTransformer, Transformer, PositionalEncodingSuperPixel

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
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.matmul(h, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)
        e = self._prepare_attentional_mechanism_input(Wh)

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # broadcast add
        e = Wh1 + Wh2.permute(0, 2, 1)
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, dropout, alpha, nheads, seq_len):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attention1 = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attention1):
            self.add_module('attention_1_{}'.format(i), attention)
        self.maxpool1 = MaxPoolingAggregator(nhid*nheads, nhid*nheads, nhid*nheads, seq_len, dropout, bias=True)

        self.attention2 = [GraphAttentionLayer(nhid*nheads, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attention2):
            self.add_module('attention_2_{}'.format(i), attention)
        self.maxpool2 = MaxPoolingAggregator(nhid*nheads, nhid*nheads, nhid*nheads, seq_len, dropout, bias=True)

        self.attention3 = [GraphAttentionLayer(nhid*nheads, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attention3):
            self.add_module('attention_3_{}'.format(i), attention)
        self.maxpool3 = MaxPoolingAggregator(nhid*nheads, nhid*nheads, nhid*nheads, seq_len, dropout, bias=True)
        # self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)
        self.transformer = Transformer(nhid * nheads, 3, nheads, nhid , nheads*nhid, dropout)
        self.out = nn.Linear(nhid * nheads, 1)
    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attention1], dim=2)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.maxpool1(x, adj)
        
        x = torch.cat([att(x, adj) for att in self.attention2], dim=2)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.maxpool2(x, adj)

        x = torch.cat([att(x, adj) for att in self.attention3], dim=2)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.maxpool3(x, adj)
        # x = F.elu(self.out_att(x, adj))
        x = self.transformer(x)
        x = self.out(x)
        return x

class GATFCN(nn.Module):
    def __init__(self, nfeat, nhid, dropout, alpha, nheads, seq_len):
        """Dense version of GAT."""
        super(GATFCN, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out = nn.Linear(nhid * nheads * seq_len, seq_len)
    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=2)
        x = F.dropout(x, self.dropout, training=self.training)

        x = x.reshape(x.size(0), -1)
        x = self.out(x)
        return x


class GATSepFCN(nn.Module):
    def __init__(self, nfeat, nhid, dropout, alpha, nheads, seq_len):
        """Dense version of GAT."""
        super(GATSepFCN, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out1 = nn.Linear(seq_len, seq_len)
        self.out2 = nn.Linear(nhid * nheads, 1)
    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=2)
        x = F.dropout(x, self.dropout, training=self.training)

        x = x.permute(0, 2, 1) # batch, C, Nodes
        x = self.out1(x)
        x = x.permute(0, 2, 1) # batch, Nodes, C
        x = self.out2(x)
        return x


class DeepGAT(nn.Module):
    def __init__(self, nfeat, nhid, block_depth, dropout, nheads, ntfm, norm='ln'):
        """Dense version of GAT."""
        super(DeepGAT, self).__init__()
        self.linear = nn.Linear(nfeat, nhid * nheads)
        self.transformers = nn.ModuleList([GraphConvTransformer(nhid*nheads, block_depth, nheads, nhid, nheads*nhid, norm=norm, dropout=dropout) for _ in range(ntfm)])
        self.pos_encoding = PositionalEncodingSuperPixel(nhid*nheads)
        self.out = nn.Linear(nhid * nheads, 1)
    def forward(self, x, adj):
        x = self.linear(x)
        for layer in self.transformers:
            x = self.pos_encoding(x)
            x = layer(x, adj)
        x = self.out(x)
        return x