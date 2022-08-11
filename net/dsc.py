import torch.nn as nn
from net.blocks import ConvBlock, SuperConvBlock
from net.blocks import ToSLIC
import torch
import torchvision.transforms as T



class DSC(nn.Module):
    def __init__(self, num_seg = 256):
        super().__init__()
        self.conv1 = nn.Sequential(*[ConvBlock(3, 64, 3), ConvBlock(64, 64, 3)])
        self.conv2 = ConvBlock(64, 32, 3)
        self.sp = ToSLIC(channels=32, n_segments=num_seg, compactness=0.1, max_num_iter=10, enforce_connectivity=False, min_size_factor=0.,)

        self.sconv1 = nn.ModuleList([SuperConvBlock(32, 8, 32, 1, 9, 255), SuperConvBlock(32, 8, 32, 1, 9, 255), SuperConvBlock(32, 8, 32, 1, 9, 255)])
        self.sconv2 = nn.ModuleList([SuperConvBlock(32, 16, 64, 2, 9, 255), SuperConvBlock(64, 16, 64, 2, 9, 255), SuperConvBlock(64, 16, 64, 2, 9, 255)])
        self.sconv3 = nn.ModuleList([SuperConvBlock(64, 32, 128, 4, 9, 255), SuperConvBlock(128, 32, 128, 4, 9, 255), SuperConvBlock(128, 32, 128, 4, 9, 255)])

        self.conv3 = nn.Sequential(*[nn.Linear(128, 32), nn.ReLU()])



    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        with torch.no_grad():
            x_sp, x_neighbours, x_labels = self.sp(x)
 
        for l in self.sconv1:
            x_sp = l(x_sp, x_neighbours)

        for l in self.sconv2:
            x_sp = l(x_sp, x_neighbours)

        for l in self.sconv3:
            x_sp = l(x_sp, x_neighbours)
    



        assert(0)

