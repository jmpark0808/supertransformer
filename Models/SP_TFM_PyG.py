import torch.nn as nn
from Blocks.GraphBlocks import *
from Blocks.TransformerBlocks import *
from dataset.constants import *
from torch_geometric.nn.conv import TransformerConv
from torch_geometric.nn.norm import LayerNorm
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
        self.elu = nn.ELU()
        self.pos_linear = nn.Linear(2, nhid*nheads)
        self.convs = nn.ModuleList([TransformerConv(in_channels=nhid*nheads, out_channels=nhid,
                                                                          heads=nheads, dropout=dropout, edge_dim=2,
                                                                            concat=True, root_weight=False) for _ in range(ntfm)])
        self.ffs = nn.ModuleList([FeedForward(nhid*nheads, nhid*nheads, dropout) for _ in range(ntfm)])
        self.ln1s = nn.ModuleList([LayerNorm(nhid*nheads) for _ in range(ntfm)])
        self.ln2s = nn.ModuleList([LayerNorm(nhid*nheads) for _ in range(ntfm)])
        self.classifier = nn.Linear(nhid*nheads, 1)
        self.num_seg = num_seg
        


        
    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        batch_size = x.size(0)//self.num_seg
        batch_index = torch.arange(0, batch_size).repeat(self.num_seg).reshape(self.num_seg, -1).T.reshape(-1).cuda()
        pos = x[:, :2]
        x = x[:, 2:]

        x = self.linear1(x)

        pos = self.pos_linear(pos)
        x += pos

        for conv, ff, ln1, ln2 in zip(self.convs, self.ffs, self.ln1s, self.ln2s):
            x = conv(ln1(x, batch_index), edge_index=edge_index, edge_attr=None) + x# adding edge features here
            x = ff(ln2(x, batch_index)) + x
        # for conv, ln1 in zip(self.convs, self.ln1s):
        #     x = ln1(x, batch_index)
        #     x = conv(x, edge_index=edge_index, edge_attr=None)# adding edge features here
        #     x = self.elu(x)
      
        x = self.classifier(x)
        return x