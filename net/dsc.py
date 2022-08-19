import torch.nn as nn
from net.blocks import ConvBlock, SuperConvBlock
from net.blocks import ToSLIC
import torch
import torchvision.transforms as T
import numpy as np


class DSC(nn.Module):
    def __init__(self, num_seg = 256):
        super().__init__()
        self.conv1 = nn.Sequential(*[ConvBlock(3, 64, 3), ConvBlock(64, 64, 3)])
        self.conv2 = ConvBlock(64, 32, 3)
        self.sp = ToSLIC(channels=32, n_segments=num_seg, compactness=0.1, max_num_iter=10, 
        enforce_connectivity=False, min_size_factor=0., max_size_factor=10)

        self.sconv1 = nn.ModuleList([SuperConvBlock(32, 8, 32, 1, 256, True), SuperConvBlock(32, 8, 32, 1, 256, True), SuperConvBlock(32, 8, 32, 1, 256, True)])
        self.sconv2 = nn.ModuleList([SuperConvBlock(32, 16, 64, 2, 256, False), SuperConvBlock(64, 16, 64, 2,  256, True), SuperConvBlock(64, 16, 64, 2, 256, True)])
        self.sconv3 = nn.ModuleList([SuperConvBlock(64, 32, 128, 4, 256, False), SuperConvBlock(128, 32, 128, 4,  256, True), SuperConvBlock(128, 32, 128, 4, 256, True)])

        self.conv3 = nn.Sequential(*[nn.Linear(128, 32), nn.ReLU()])

        self.conv4 = nn.Sequential(*[nn.Linear(64, 64), nn.ReLU()])
        self.conv5 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.conv1(x)
        print(sum(torch.norm(p) for p in self.conv1.parameters()), sum(torch.norm(p) for p in self.conv2.parameters()),
        sum(torch.norm(p) for p in self.sconv1.parameters()), sum(torch.norm(p) for p in self.sconv2.parameters()), sum(torch.norm(p) for p in self.sconv3.parameters()),
        sum(torch.norm(p) for p in self.conv3.parameters()), sum(torch.norm(p) for p in self.conv4.parameters()), sum(torch.norm(p) for p in self.conv5.parameters()))
        x = self.conv2(x)
 
        
        x_sp, x_neighbours, x_labels = self.sp(x)
        x_sp = x_sp.cuda()
        x_neighbours = x_neighbours.cuda()
 
        for l in self.sconv1:
            x_sp = l(x_sp, x_neighbours)
        # print(dict(self.sconv1.named_parameters()).keys())

        for name, param in self.named_parameters():
             if param.requires_grad:
                 print(name, param.grad)

        # print(sum(torch.norm(p) for p in self.sconv1.parameters()))
        # assert(0)
        for l in self.sconv2:
            x_sp = l(x_sp, x_neighbours)

        for l in self.sconv3:
            x_sp = l(x_sp, x_neighbours)

        x_sp = self.conv3(x_sp)

        
        segments = x_labels.reshape([x_labels.shape[0], -1]) # batch, img_size^2

        samples = []
        for masked, labels in zip(x_sp.detach().cpu().numpy(), segments):
            plt_image = masked[labels-1].reshape([256, 256, 32])
            samples.append(plt_image)

        samples = torch.tensor(np.array(samples)) # B, 256, 256, 32
        samples = samples.permute(0, 3, 1, 2) # B, 32, 256, 256
        samples = samples.cuda()

        concat = torch.cat([x, samples], dim=1) # B, 64, 256, 256
        concat = concat.permute(0, 2, 3, 1)

        conv4 = self.conv4(concat)

        conv5 = self.conv5(conv4)
        conv5 = conv5.permute(0, 3, 1, 2) # B, 1, 256, 256

        return conv5




