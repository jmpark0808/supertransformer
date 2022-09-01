import torch.nn as nn
from Blocks.blocks import SuperConvBlock

class SP_SCB_DIL(nn.Module):
    '''
    Graph Convolutions using generalized graph convolution + global aggregation using dilations
    no positional encoding
    '''
    def __init__(self, num_seg):
        """Dense version of GAT."""
        self.num_seg = num_seg
        super(SP_SCB_DIL, self).__init__()
        self.sconv1 = nn.ModuleList([SuperConvBlock(9, 8, 32, 1, self.num_seg, False, False), SuperConvBlock(32, 8, 32, 1, self.num_seg, True, False),
         SuperConvBlock(32, 8, 32, 1, self.num_seg, True, False)])
        self.sconv2 = nn.ModuleList([SuperConvBlock(32, 16, 64, 2, self.num_seg, False, False), SuperConvBlock(64, 16, 64, 2,  self.num_seg, True, False),
         SuperConvBlock(64, 16, 64, 2, self.num_seg, True, False)])
        self.sconv3 = nn.ModuleList([SuperConvBlock(64, 32, 128, 4, self.num_seg, False, False), SuperConvBlock(128, 32, 128, 4,  self.num_seg, True, False),
         SuperConvBlock(128, 32, 128, 4, self.num_seg, True, False)])
        self.linear = nn.Linear(128, 1)
    def forward(self, x, adj):
        x = x[:, :, 2:]
        for l in self.sconv1:
            x = l(x, adj)

        for l in self.sconv2:
            x = l(x, adj)

        for l in self.sconv3:
            x = l(x, adj)

        
        pred = self.linear(x)
        return pred





