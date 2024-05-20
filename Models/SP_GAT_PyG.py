import torch.nn as nn
from Blocks.GraphBlocks import *
from Wrappers.PositionalEncoding import PositionalEncodingSuperPixel
from Blocks.TransformerBlocks import *
from dataset.constants import *
from torch_geometric.nn.conv import GATv2Conv
from torch_geometric.nn.norm import GraphNorm, LayerNorm
from torch.nn import LayerNorm as TLayerNorm
from torch_geometric.nn.pool import global_mean_pool
from torch_geometric.nn.dense import DenseGATConv
from torch_sparse import SparseTensor
from torch_geometric.utils import dropout_edge
from torch_geometric.nn.pool import TopKPooling


class SP_GAT_PyG(nn.Module):
    '''
    Pure Global aggregation using transformers
    Deterministic Positional Encoding 
    '''
    def __init__(self, nfeat, nhid, edge_dim, dropout, nheads, ntfm, num_seg, dilation_mode, dilation):
        """Dense version of GAT."""
        super(SP_GAT_PyG, self).__init__()
        self.linear1 = nn.Linear(nfeat, nhid)
        self.elu = nn.ReLU()
        self.pos_linear = nn.Linear(2, nhid)
        self.convs = nn.ModuleList([GATv2Conv(in_channels=nhid, out_channels=nhid,
                                                                          heads=nheads, dropout=dropout, edge_dim=None,
                                                                            concat=False) for _ in range(ntfm)])
        
        self.ln1s = nn.ModuleList([GraphNorm(nhid) for _ in range(ntfm)])

        self.classifier = nn.Linear(nhid, 1)
        self.num_seg = num_seg
        self.dilation_mode = dilation_mode
        self.dilation = dilation
        


        
    def forward(self, data, return_attention=False):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        batch_size = x.size(0)//self.num_seg
        batch_index = torch.arange(0, batch_size).repeat(self.num_seg).reshape(self.num_seg, -1).T.reshape(-1).cuda()
        pos = x[:, :2]
        x = x[:, 2:]

        x = self.linear1(x)
        
        pos = self.pos_linear(pos)
        x += pos

        att_weights = []
        for ln, conv in zip(self.ln1s, self.convs):
            h = x
            if return_attention:
                x, (ei, att) = conv(x, edge_index=edge_index, edge_attr=None, return_attention_weights=return_attention)# adding edge features here
                att_weights.append(att)
            else:
                
                x = conv(x, edge_index=edge_index, edge_attr=None)
            x = ln(x, batch=batch_index)
            x = self.elu(x) + h
        # for conv in self.convs:
        #     if return_attention:
        #         x, (ei, att) = conv(x, edge_index=edge_index, edge_attr=None, return_attention_weights=return_attention)# adding edge features here
        #         att_weights.append(att)
        #     else:
        #         x = conv(x, edge_index=edge_index, edge_attr=None)# adding edge features here
            
        #     x = self.elu(x)

      
        # x = self.convs[-1](x, edge_index, edge_attr=edge_attr)
        x = self.classifier(x)
        if return_attention:
            return x, att_weights
        else:
            return x
    

class SP_GAT_IN(nn.Module):
    '''
    Pure Global aggregation using transformers
    Deterministic Positional Encoding 
    '''
    def __init__(self, nfeat, nhid, dropout, dropout_edge, nheads, ntfm, num_seg):
        """Dense version of GAT."""
        super(SP_GAT_IN, self).__init__()
        self.linear1 = nn.Linear(nfeat, nhid)
        self.ln1 = LayerNorm(nhid)
        self.relu = nn.ReLU()
        self.convs = nn.ModuleList([GATv2Conv(in_channels=nhid, out_channels=nhid,
                                                                          heads=nheads, dropout=dropout_edge, edge_dim=2,
                                                                            concat=False) for _ in range(ntfm)])
        
        self.lns = nn.ModuleList([LayerNorm(nhid, mode='node') for _ in range(ntfm)])
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(nhid, 1000)
        self.num_seg = num_seg
        


        
    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
       
        batch_size = x.size(0)//self.num_seg
        batch_index = torch.arange(0, batch_size).repeat(self.num_seg).reshape(self.num_seg, -1).T.reshape(-1).cuda()
        x = x[:, 2:]

        x = self.linear1(x)
        x = self.ln1(x)
        x = self.relu(x)

        
        for conv, ln in zip(self.convs, self.lns):
            h = x
            x = ln(x, batch_index)
            x = conv(x, edge_index=edge_index, edge_attr=edge_attr)# adding edge features here
            x = self.relu(x)
            x = h + x

      
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

