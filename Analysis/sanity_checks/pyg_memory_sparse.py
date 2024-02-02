import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch.nn as nn
from torch_geometric.nn.conv import GATv2Conv
from torch_geometric.nn.conv import SAGEConv
from torch_sparse import SparseTensor
import torch_geometric.transforms as T
use_amp = False


class DummyModelSparse(nn.Module):
    '''
    Pure Global aggregation using transformers
    Deterministic Positional Encoding 
    '''
    def __init__(self, nfeat, nhid, dropout, nheads, nlayer):
        """Dense version of GAT."""
        super(DummyModelSparse, self).__init__()
        self.linear = nn.Linear(nfeat, nhid)
        # self.convs = nn.ModuleList([GATv2Conv(in_channels=nhid, out_channels=nhid,
        #                                                                   heads=nheads, dropout=dropout, edge_dim=None,
        #                                                                     concat=False) for _ in range(nlayer)])
        self.convs = nn.ModuleList([SAGEConv(in_channels=nhid, out_channels=nhid) for _ in range(nlayer)])
    def forward(self, data):
        x, edge_index = data.x, data.adj_t

        x = self.linear(x)
        for conv in self.convs:
            x = conv(x, edge_index=edge_index)
            
        return x
    

Sparse_model = DummyModelSparse(26,16, 0, 8, 6)

device = 0
optimizer_type = torch.optim.Adam
Sparse_model.cpu()



optimizer = optimizer_type(Sparse_model.parameters(), lr=.001)
a = torch.cuda.memory_allocated(device)
Sparse_model.to(device)
b = torch.cuda.memory_allocated(device)
model_memory = b - a
print(f'Sparse Model memory: {model_memory/1e9}')

torch.manual_seed(0)
adj = torch.empty(625, 625).uniform_(0, 1)
edge_index = torch.bernoulli(adj)
edge_index = edge_index.nonzero().t().contiguous()
print(edge_index.size())

dummy_features = torch.randn(625, 26)

data = Data(x=dummy_features, edge_index=edge_index)
data_list = [data]*4
loader = DataLoader(data_list, batch_size=4)
batch = next(iter(loader))

transform = T.Compose([T.ToSparseTensor()])
batch = transform(batch)


output = Sparse_model(batch.to(device))
c = torch.cuda.memory_allocated(device)

if use_amp:
    amp_multiplier = .5
else:
    amp_multiplier = 1
forward_pass_memory = (c - b)*amp_multiplier
print(f'Sparse Forward pass memory: {forward_pass_memory/1e9}')
gradient_memory = model_memory
if isinstance(optimizer, torch.optim.Adam):
    o = 2
elif isinstance(optimizer, torch.optim.RMSprop):
    o = 1
elif isinstance(optimizer, torch.optim.SGD):
    o = 0
elif isinstance(optimizer, torch.optim.Adagrad):
    o = 1
else:
    raise ValueError("Unsupported optimizer. Look up how many moments are" +
        "stored by your optimizer and add a case to the optimizer checker.")
gradient_moment_memory = o*gradient_memory
total_memory = model_memory + forward_pass_memory + gradient_memory + gradient_moment_memory 
print(f'Sparse Total memory: {total_memory/1e9}')







