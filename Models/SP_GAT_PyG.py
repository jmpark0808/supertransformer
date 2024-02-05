import torch.nn as nn
from Blocks.GraphBlocks import *
from Wrappers.PositionalEncoding import PositionalEncodingSuperPixel
from Blocks.TransformerBlocks import *
from dataset.constants import *
from torch_geometric.nn.conv import GATv2Conv
from torch_geometric.nn.norm import LayerNorm
from torch.nn import LayerNorm as TLayerNorm
from torch_geometric.nn.pool import global_mean_pool
from torch_geometric.nn.dense import DenseGATConv
from torch_sparse import SparseTensor



class SP_GAT_PyG(nn.Module):
    '''
    Pure Global aggregation using transformers
    Deterministic Positional Encoding 
    '''
    def __init__(self, nfeat, nhid, edge_dim, dropout, nheads, ntfm, num_seg):
        """Dense version of GAT."""
        super(SP_GAT_PyG, self).__init__()
        self.linear1 = nn.Linear(nfeat, nhid)
        self.elu = nn.ReLU()
        self.pos_linear = nn.Linear(2, nhid)
        self.convs = nn.ModuleList([GATv2Conv(in_channels=nhid, out_channels=nhid,
                                                                          heads=nheads, dropout=dropout, edge_dim=None,
                                                                            concat=False) for _ in range(ntfm)])
        
        # self.ln1s = nn.ModuleList([LayerNorm(nhid, mode='node') for _ in range(ntfm)])

        self.classifier = nn.Linear(nhid, 1)
        self.num_seg = num_seg
        


        
    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        # batch_size = x.size(0)//self.num_seg
        # batch_index = torch.arange(0, batch_size).repeat(self.num_seg).reshape(self.num_seg, -1).T.reshape(-1).cuda()
        pos = x[:, :2]
        x = x[:, 2:]

        x = self.linear1(x)
        
        pos = self.pos_linear(pos)
        x += pos

        for conv in self.convs:
            x = conv(x, edge_index=edge_index, edge_attr=None)# adding edge features here
            x = self.elu(x)

      
        # x = self.convs[-1](x, edge_index, edge_attr=edge_attr)
        x = self.classifier(x)
        return x
    

class SP_GAT_IN(nn.Module):
    '''
    Pure Global aggregation using transformers
    Deterministic Positional Encoding 
    '''
    def __init__(self, nfeat, nhid, dropout, nheads, ntfm, num_seg):
        """Dense version of GAT."""
        super(SP_GAT_IN, self).__init__()
        self.linear1 = nn.Linear(nfeat, nhid*nheads)
        self.elu = nn.ELU()
        self.pos_linear = nn.Linear(2, nhid*nheads)
        self.convs = nn.ModuleList([GATv2Conv(in_channels=nhid*nheads, out_channels=nhid,
                                                                          heads=nheads, dropout=dropout, edge_dim=None,
                                                                            concat=True) for _ in range(ntfm)])
        
        self.ln1s = nn.ModuleList([LayerNorm(nhid*nheads) for _ in range(ntfm)])
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(nhid*nheads, 1000)
        self.num_seg = num_seg
        


        
    def forward(self, data):
        x, edge_index = data.x, data.adj_t
        # edge_index = SparseTensor(row=edge_index[0], col=edge_index[1]).t()
       
        batch_size = x.size(0)//self.num_seg
        batch_index = torch.arange(0, batch_size).repeat(self.num_seg).reshape(self.num_seg, -1).T.reshape(-1).cuda()
        pos = x[:, :2]
        x = x[:, 2:]

        x = self.linear1(x)
        x = self.elu(x)

        pos = self.pos_linear(pos)
        x += pos
        
        for conv, ln1 in zip(self.convs, self.ln1s):
            x = ln1(x, batch_index)
            x = conv(x, edge_index=edge_index, edge_attr=None)# adding edge features here
            x = self.elu(x)

      
        x = global_mean_pool(x, batch_index)
        x = self.dropout(x)
        x = self.classifier(x)
        return x
    

class SP_GAT_DIN(nn.Module):
    '''
    Pure Global aggregation using transformers
    Deterministic Positional Encoding 
    '''
    def __init__(self, nfeat, nhid, dropout, nheads, ntfm, num_seg):
        """Dense version of GAT."""
        super(SP_GAT_DIN, self).__init__()
        self.linear1 = nn.Linear(nfeat, nhid)
        self.elu = nn.ELU()
        self.pos_linear = nn.Linear(2, nhid)
        self.convs = nn.ModuleList([DenseGATConv(nhid, nhid, nheads, False, 0.2, dropout) for _ in range(ntfm)])
        
        self.ln1s = nn.ModuleList([TLayerNorm(nhid) for _ in range(ntfm)])
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(nhid, 1000)
        self.num_seg = num_seg
        


        
    def forward(self, x, adj):
        pos = x[:, :, :2]
        x = x[:, :, 2:]

        x = self.linear1(x)
        x = self.elu(x)

        pos = self.pos_linear(pos)
        x += pos
        
        for conv, ln1 in zip(self.convs, self.ln1s):
            x = ln1(x)
            x = conv(x, adj)# adding edge features here
            x = self.elu(x)

      
        x = torch.mean(x, dim=1)
        x = self.dropout(x)
        x = self.classifier(x)
        return x
# class SP_GAT_PyG(nn.Module):
#     '''
#     Pure Global aggregation using transformers
#     Deterministic Positional Encoding 
#     '''
#     def __init__(self, nfeat, nhid, edge_dim, dropout, nheads, ntfm):
#         """Dense version of GAT."""
#         super(SP_GAT_PyG, self).__init__()
#         self.linear1 = nn.Linear(nfeat, nhid)
#         self.elu = nn.ELU()
    
#         self.convs = nn.ModuleList([GATv2Conv(in_channels=nhid, out_channels=nhid, heads=nheads, dropout=dropout, edge_dim=edge_dim, concat=False) for _ in range(ntfm)])
#         self.classifier = nn.Linear(nhid, 1)


        
#     def forward(self, data):
#         x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

#         x = self.linear1(x)
#         x = self.elu(x)

#         for conv in self.convs[:-1]:
#             x = conv(x, edge_index, edge_attr=edge_attr) # adding edge features here!
#             x = self.elu(x)

      
#         x = self.convs[-1](x, edge_index, edge_attr=edge_attr)
#         x = self.classifier(x)
#         return x

