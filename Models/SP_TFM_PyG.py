import torch.nn as nn
from Blocks.GraphBlocks import *
from Blocks.TransformerBlocks import *
from dataset.constants import *
from Blocks.GraphTransformer import TransformerConv
from torch_geometric.nn.norm import GraphNorm
from Blocks.TransformerBlocks import FeedForward 


class SP_TFM_PyG(nn.Module):
    '''
    Pure Global aggregation using transformers
    Deterministic Positional Encoding 
    '''
    def __init__(self, nfeat, nhid, edge_dim, dropout, nheads, ntfm, num_seg):
        """Dense version of GAT."""
        super(SP_TFM_PyG, self).__init__()
        self.linear1 = nn.Linear(nfeat, nhid*nheads)
        self.elu = nn.ReLU()
        self.pos_linear = nn.Linear(2, nhid*nheads)
        self.convs = nn.ModuleList([TransformerConv(in_channels=nhid*nheads, out_channels=nhid,
                                                                          heads=nheads, dropout=dropout, edge_dim=None,
                                                                            concat=True, root_weight=False) for _ in range(ntfm)])
        self.ln1s = nn.ModuleList([GraphNorm(nhid*nheads) for _ in range(ntfm)])
        self.classifier = nn.Linear(nhid*nheads, 1)
        self.num_seg = num_seg
        


        
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