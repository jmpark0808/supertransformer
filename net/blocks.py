from tkinter import Label
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
from skimage.segmentation import slic
from net.gcn import MaxPoolingCNN
from skimage.measure import regionprops_table


def make_layers(cfg, in_channels):
    layers = []
    dilation_flag = False
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'm':
            layers += [nn.MaxPool2d(kernel_size=1, stride=1)]
            dilation_flag = True
        elif v == 'p':
            layers += [nn.MaxPool2d(kernel_size=3, stride=1, padding=1)]
        else:
            if not dilation_flag:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=2, dilation=2)
            layers += [conv2d, nn.ReLU()]
            in_channels = v
    return nn.Sequential(*layers)

def make_layers_bn(cfg, in_channels):
    layers = []
    dilation_flag = False
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'm':
            layers += [nn.MaxPool2d(kernel_size=1, stride=1)]
            dilation_flag = True
        elif v == 'p':
            layers += [MaxPoolingCNN(28, in_channels, in_channels, in_channels, 3, 1, dropout=1, bias=True)]
        else:
            if not dilation_flag:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=2, dilation=2)
            layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU()]
            in_channels = v
    return nn.Sequential(*layers)


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        configure = [64, 64, 'p', 128, 128, 'p', 256, 256, 256, 'p', 512, 512, 512, 'm', 512, 512, 512, 'm']
        self.seq = make_layers(configure, 3)


    def forward(self, x):
        conv1 = self.seq(x)

        return conv1

class EncoderNMP(nn.Module):
    def __init__(self):
        super(EncoderNMP, self).__init__()
        configure = [64, 64, 'p', 128, 128, 'p', 256, 256, 256, 'p', 512, 512, 512, 'p', 512, 512, 512, 'p']
        self.seq = make_layers_bn(configure, 3)


    def forward(self, *input):
        x = input[0]
        conv1 = self.seq(x)
  

        return conv1

class EncoderDilated(nn.Module):
    def __init__(self, resolution, kernel):
        super(EncoderDilated, self).__init__()
        channels = [3, 64, 128, 256, 512, 512, 512, 512]
        dilations = [1, 2, 4, 8, 4, 2, 1]
        layers = []
        for i, o, d in zip(channels[:-1], channels[1:], dilations):
            padding = ((resolution-1)*1-resolution+3+(3-1)*(d-1))/2
            layers += [nn.Conv2d(i, o, kernel_size=kernel, stride=1, padding=int(padding), dilation=d), nn.BatchNorm2d(o), nn.ReLU()]
        self.seq = nn.Sequential(*layers)


    def forward(self, x):
        conv1 = self.seq(x)

        return conv1


class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size):
        super().__init__()
        if kernel_size == 3:
            padding = 1
        elif kernel_size == 1:
            padding = 0
        else:
            raise('Not a supported kernel size')

        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, padding=padding, stride=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)       
        x = self.relu(x)
        return x

class ToSLIC(nn.Module):
    def __init__(self, channels=32, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.channels = 32

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        b, h, w, c = x.size()

        all_features = []
        all_neighbours = []
        all_labels = []
        for one_x in x:
            segments = slic(one_x.to(torch.double).numpy(), start_label=0, **self.kwargs)
   
            vs_right = np.vstack([segments[:,:-1].ravel(), segments[:,1:].ravel()])
            vs_below = np.vstack([segments[:-1,:].ravel(), segments[1:,:].ravel()])
            bneighbors = np.unique(np.hstack([vs_right, vs_below]), axis=1)
            
            regions = regionprops_table(segments, intensity_image=one_x.to(torch.double).numpy(), properties=('label', 'intensity_max'))
            seq_len = len(regions['label'])
            neighbor_array = np.zeros([seq_len, seq_len])
            neighbor_array[bneighbors[0]-1, bneighbors[1]-1] = 1
            label = regions['label']
            features = np.zeros([seq_len, self.channels])
            for i in range(self.channels):
                features[label-1, i] = regions[f'intensity_max-{i}']

            all_features.append(features)
            all_neighbours.append(neighbor_array)
            all_labels.append(label)

        all_features = np.stack(all_features, axis=0)
        all_neighbours = np.stack(all_neighbours, axis=0)
        all_labels = np.stack(all_labels, axis=0)

        return torch.from_numpy(all_features).float(), torch.from_numpy(all_neighbours).float(), all_labels

class SuperConvBlock(nn.Module):
    def __init__(self, in_channel, mid_channel, out_channel, dilation, p, num_regions):
        super().__init__()
        self.dilation = dilation
        self.eye = torch.eye(num_regions)

        self.conv1x1_1 = nn.Linear(in_channel, mid_channel)
        self.tanh_1 = nn.Tanh()

        self.W1 = nn.Parameter(torch.randn(mid_channel, mid_channel, p))
        self.W2 = nn.Parameter(torch.randn(p, num_regions))

        self.conv1x1_2 = nn.Linear(mid_channel, out_channel)
        self.tanh_2 = nn.Tanh()

    def circulant(self, tensor, dim):
        """get a circulant version of the tensor along the {dim} dimension.
        
        The additional axis is appended as the last dimension.
        E.g. tensor=[0,1,2], dim=0 --> [[0,1,2],[2,0,1],[1,2,0]]"""
        S = tensor.shape[dim]
        tmp = torch.cat([tensor.flip((dim,)), torch.narrow(tensor.flip((dim,)), dim=dim, start=0, length=S-1)], dim=dim)
        return tmp.unfold(dim, S, 1).flip((-1,))

    def forward(self, x, A):
        conv1 = self.tanh_1(self.conv1x1_1(x)) # B x R x C

        circulant = self.circulant(self.W2, 1) # p x R x R
        adj = torch.matrix_power(A, self.dilation)-torch.matrix_power(A, self.dilation-1)+self.eye
        adj = adj.unsqueeze(0)
        circulant = circulant*adj
        circulant = torch.sum(circulant, dim=-1) # p x R

        W = torch.matmul(self.W1, circulant).unsqueeze(0).repeat(x.size(0), 1, 1, 1) # B x C' x C x R
        conv1 = conv1.unsqueeze(2) # B x R x 1 x C
        conv1 = conv1.permute(0, 3, 2, 1) # B x C x 1 x R

        x = torch.einsum('bijk,bjlk->bilk',W, conv1) # B x C' x 1 x R
        x = x.reshape(x.size(0), -1, x.size(3)).permute(0, 2, 1) # B x R x C'

        conv2 = self.tanh_2(self.conv1x1_2(x)) # B x R x C"
        return conv2

        



