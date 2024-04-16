import torch.nn as nn
from Blocks.GraphBlocks import *
from Blocks.TransformerBlocks import *
from dataset.constants import *
from Blocks.GraphTransformer import TransformerConv
from torch_geometric.nn.norm import GraphNorm
from Blocks.TransformerBlocks import FeedForward 


class SP_SWIN_PyG(nn.Module):
    '''
    Pure Global aggregation using transformers
    Deterministic Positional Encoding 
    '''
    def __init__(self, nfeat, nhid, head_dim, edge_dim, dropout, nheads, ntfm, num_seg, window_size):
        """Dense version of GAT."""
        super().__init__()
        self.linear1 = nn.Linear(nfeat, nhid)
        self.elu = nn.ReLU()
        # self.pos_linear = nn.Linear(2, nhid)
        assert ntfm%3==0, 'NTFM must be divisible by 3'
        self.convs = nn.ModuleList([TransformerConv(in_channels=nhid, out_channels=head_dim,
                                                                          heads=nheads, dropout=dropout, edge_dim=edge_dim,
                                                                            concat=False, root_weight=False) for _ in range(ntfm)])
        self.ln1s = nn.ModuleList([GraphNorm(nhid) for _ in range(ntfm)])
        self.classifier = nn.Linear(nhid, 1)
        self.window_size = window_size
        self.num_seg = num_seg
        


        
    def forward(self, data, return_attention=False):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        num_edges = (self.window_size**4)*((32//self.window_size)**2)
        edge_index = edge_index.reshape(2, num_edges, 3, -1)
        dilated_index = edge_index[:, :, 0, :].reshape(2, -1)
        local_index = edge_index[:, :, 1, :].reshape(2, -1)
        shifted_index = edge_index[:, :, 2, :].reshape(2, -1)

        edge_attr_dim = edge_attr.size(1)
        edge_attr = edge_attr.reshape( num_edges, 3, -1, edge_attr_dim)
        dilated_edge_attr = edge_attr[:, 0, :, :].reshape( -1, edge_attr_dim)
        local_edge_attr = edge_attr[:, 1, :, :].reshape( -1, edge_attr_dim)
        shifted_edge_attr = edge_attr[:, 2, :, :].reshape( -1, edge_attr_dim)
        
        batch_size = x.size(0)//self.num_seg
        batch_index = torch.arange(0, batch_size).repeat(self.num_seg).reshape(self.num_seg, -1).T.reshape(-1).cuda()
        pos = x[:, :2]
        x = x[:, 2:]

        x = self.linear1(x)
        
        # pos = self.pos_linear(pos)
        # x += pos

        att_weights = []
        for idx, (ln, conv) in enumerate(zip(self.ln1s, self.convs)):
            if idx % 3 == 0:
                edge_index = local_index
                edge_attr = local_edge_attr
            elif idx % 3 == 1:
                edge_index = shifted_index
                edge_attr = shifted_edge_attr
            else:
                edge_index = dilated_index
                edge_attr = dilated_edge_attr

            h = x
            if return_attention:
                x, (ei, att) = conv(x, edge_index=edge_index, edge_attr=edge_attr, return_attention_weights=return_attention)# adding edge features here
                att_weights.append(att)
            else:
                x = conv(x, edge_index=edge_index, edge_attr=edge_attr)
           
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