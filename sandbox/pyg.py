import torch
from torch_geometric.nn.conv import GATv2Conv
import torch.functional as F
import torch.nn as nn

class BaselineGDPModel(torch.nn.Module):
    def __init__(self, num_features=3, hidden_size=32, target_size=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_features = num_features
        self.target_size = target_size
        self.convs = [GATv2Conv(self.num_features, self.hidden_size),
                      GATv2Conv(self.hidden_size, self.hidden_size)]
        self.linear = nn.Linear(self.hidden_size, self.target_size)


    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        for conv in self.convs[:-1]: 
            x = conv(x, edge_index)
            x = torch.relu(x)
            x = torch.dropout(x, 0.5, train=self.training)
        x = self.convs[-1](x, edge_index) 
        x = self.linear(x)
        return torch.relu(x) 
    
class GDPModel(torch.nn.Module):
    def __init__(self, num_features=3, hidden_size=32, target_size=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_features = num_features
        self.target_size = target_size
        self.convs = [GATv2Conv(self.num_features, self.hidden_size, edge_dim = 3),
                      GATv2Conv(self.hidden_size, self.hidden_size, edge_dim = 3)]
        self.linear = nn.Linear(self.hidden_size, self.target_size)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        for conv in self.convs[:-1]:
            x = conv(x, edge_index, edge_attr=edge_attr) # adding edge features here!
            x = F.relu(x)
            x = F.dropout(x, training=self.training)
        x = self.convs[-1](x, edge_index, edge_attr=edge_attr) # edge features here as well
        x = self.linear(x)

        return F.relu(x) 

adj = torch.ones(40, 40)
# adj = torch.tensor([[0, 1, 1],
#                      [0, 0, 0],
#                        [1, 0, 1]])
edge_index = adj.nonzero().t().contiguous()

# print(edge_index)

# import numpy as np

# adj = np.array([[0, 1, 1],
#                      [0, 0, 0],
#                        [1, 0, 1]])
# edge_index = np.nonzero(adj)
# print(edge_index)
# assert(0)
edge_features = torch.ones(40, 40, 3)
edge_features = edge_features[adj.nonzero().t().numpy()]
t = torch.randn(40, 3)


from torch_geometric.data import Data, HeteroData
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader


import torch.utils.data as datatorch
import numpy as np
class SPDataset(datatorch.Dataset):
    def __init__(self):
        pass

    def __len__(self):
        return 100

    def __getitem__(self, idx):
        random_int = np.random.randint(30, 50)
        t = torch.randn(random_int, 3)
        adj = torch.ones(random_int, random_int)

        edge_index = adj.nonzero().t().contiguous()
        edge_features = torch.ones(random_int, random_int, 3)

        d = {'features': Data(x=t, edge_index=edge_index, edge_attr=edge_features)

        }
        return d

adj_s = torch.ones(30, 30)  

edge_index_s = adj_s.nonzero().t().contiguous()
edge_features_s = torch.ones(30, 30, 3)
edge_features_s = edge_features_s[adj_s.nonzero().t().numpy()]
s = torch.randn(30, 3)

data = Data(x=t, edge_index=edge_index, edge_attr=edge_features)
data_s = Data(x=t, edge_index=edge_index_s, edge_attr=edge_features_s)
data_list = [data, data_s]

# data_list = SPDataset()

loader = DataLoader(data_list, batch_size=2, shuffle=True)
model = BaselineGDPModel(3, 32, 1)
for d in loader:
    out = model(d)
    print(out.size())