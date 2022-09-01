from Blocks.GraphBlocks import GraphAttentionLayer, MaxPoolingAggregator
from Blocks.TransformerBlocks import Transformer
import torch.nn as nn
import torch.functional as F
import torch


class SP_GAT_TFM(nn.Module):
    '''
    Graph Convolutions using GAT + Global aggregation using transformers
    Graph Maxpooling 
    No Positional Encoding 
    '''
    def __init__(self, nfeat, nhid, dropout, alpha, nheads, seq_len):
        """Dense version of GAT."""
        super(SP_GAT_TFM, self).__init__()
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
        x = self.transformer(x)
        x = self.out(x)
        return x

