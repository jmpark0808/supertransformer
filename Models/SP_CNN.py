import torch.nn as nn
import torch

class SP_CNN_LIN(nn.Module):
    '''
    Graph Convolutions using regular convolutions
    '''
    def __init__(self):
        """Dense version of GAT."""
        super(SP_CNN_LIN, self).__init__()
        self.conv1 = nn.Conv2d(70, 128, 3, 1, 1)
        self.conv2 = nn.Conv2d(128, 128, 3, 1, 1)
        self.lin1 = nn.Conv2d(128, 128, 1, 1, 0)
        self.lin2 = nn.Conv2d(128, 1, 1, 1 ,0)

    def forward(self, x):
        '''
        x = (batch_size, 625, 70)
        '''
        x = x.reshape(x.size(0), 25, 25, 70).permute(0, 3, 1, 2)
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.lin1(x)
        x = torch.relu(x)
        x = self.lin2(x)
        return x