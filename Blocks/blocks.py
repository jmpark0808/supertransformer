from multiprocessing.forkserver import read_signed
from tkinter import Label
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
from skimage.segmentation import slic
from Blocks.GraphBlocks import MaxPoolingCNN
from skimage.measure import regionprops_table
from fast_slic.avx2 import SlicAvx2
import math

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
        self.channels = channels

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        b, h, w, c = x.size()

        all_features = []
        all_neighbours = []
        all_labels = []
        for one_x in x:
            segments = slic(one_x.to(torch.double).detach().cpu().numpy(), start_label=0, **self.kwargs)
   
            vs_right = np.vstack([segments[:,:-1].ravel(), segments[:,1:].ravel()])
            vs_below = np.vstack([segments[:-1,:].ravel(), segments[1:,:].ravel()])
            bneighbors = np.unique(np.hstack([vs_right, vs_below]), axis=1)
            
            regions = regionprops_table(segments, intensity_image=one_x.to(torch.double).detach().cpu().numpy(), properties=('label', 'intensity_max'))
            seq_len = len(regions['label'])

            neighbor_array = np.zeros([self.kwargs['n_segments'], self.kwargs['n_segments']])
            neighbor_array[bneighbors[0]-1, bneighbors[1]-1] = 1
            label = regions['label']
            features = np.zeros([self.kwargs['n_segments'], self.channels])
            for i in range(self.channels):
                features[label-1, i] = regions[f'intensity_max-{i}']

            all_features.append(features)
            all_neighbours.append(neighbor_array)
            all_labels.append(segments)

        all_features = np.stack(all_features, axis=0)
        all_neighbours = np.stack(all_neighbours, axis=0)
        all_labels = np.stack(all_labels, axis=0)

        return torch.from_numpy(all_features).float(), torch.from_numpy(all_neighbours).float(), all_labels

class SuperConvBlock(nn.Module):
    def __init__(self, in_channel, mid_channel, out_channel, dilation, num_regions, residual, separable=True):
        super().__init__()
        self.dilation = dilation
        self.eye = torch.eye(num_regions, device='cuda')

        self.conv1x1_1 = nn.Linear(in_channel, mid_channel)
        self.bn1 = nn.BatchNorm1d(mid_channel)
        self.tanh_1 = nn.ReLU()
        self.separable = separable
        if separable:
            self.W_spatial = nn.Parameter(torch.randn(mid_channel, num_regions))
            self.W_pointwise = nn.Parameter(torch.rand(mid_channel, mid_channel))
        else:
            self.W = nn.Parameter(torch.randn(mid_channel, mid_channel, num_regions))
        self.bn2 = nn.BatchNorm1d(mid_channel)
        self.tanh_2 = nn.ReLU()

        self.conv1x1_2 = nn.Linear(mid_channel, out_channel)
        self.bn3 = nn.BatchNorm1d(out_channel)
        self.tanh_3 = nn.ReLU()

        self.residual = residual
    def circulant(self, tensor, dim):
        """get a circulant version of the tensor along the {dim} dimension.
        
        The additional axis is appended as the last dimension.
        E.g. tensor=[0,1,2], dim=0 --> [[0,1,2],[2,0,1],[1,2,0]]"""
        S = tensor.shape[dim]
        tmp = torch.cat([tensor.flip((dim,)), torch.narrow(tensor.flip((dim,)), dim=dim, start=0, length=S-1)], dim=dim)
        return tmp.unfold(dim, S, 1).flip((-1,))

    def forward(self, x, A):
        # x = B x R x C , A = B x R x R
        identity = x
        conv1 = self.conv1x1_1(x) # B x R x C
        conv1 = conv1.permute(0, 2, 1) # B x C x R
        conv1 = self.tanh_1(self.bn1(conv1))
        conv1 = conv1.permute(0, 2, 1) # B x R x C

        if self.separable:
            circulant = self.circulant(self.W_spatial, 1).unsqueeze(0).repeat(x.size(0), 1, 1, 1) # B x C x R x R
            adj = torch.matrix_power(A, self.dilation).bool().int()-torch.matrix_power(A, self.dilation-1).bool().int()+self.eye
            adj = adj.unsqueeze(1).bool() # B x 1 x R x R

            circulant = circulant*adj # B x C x R x R

            conv2 = torch.einsum('bijk,bki->bij', circulant, conv1) # B x C x R
            conv2 = torch.einsum('bij,ik->bkj', conv2, self.W_pointwise) # B x C' x R
        else:
            circulant = self.circulant(self.W, 2).unsqueeze(0).repeat(x.size(0), 1, 1, 1, 1) # B x C x C' x R x R

            adj = torch.matrix_power(A, self.dilation).bool().int()-torch.matrix_power(A, self.dilation-1).bool().int()+self.eye
            adj = adj.unsqueeze(1).unsqueeze(1).bool() # B x 1 x 1 x R x R

            circulant = circulant * adj # B x C x C' x R x R



            conv2 = torch.einsum('bijkl,bli->bijk', circulant, conv1) # B x C x C' x R
            conv2 = torch.sum(conv2, dim=1) # B x C' x R 

        
        
        conv2 = self.bn2(conv2)
        conv2 = conv2.permute(0, 2, 1) # B x R x C'
        conv2 = self.tanh_2(conv2)


        conv3 = self.conv1x1_2(conv2) # B x R x C"
        conv3 = conv3.permute(0, 2, 1) # B x C" x R
        conv3 = self.bn3(conv3)
        conv3 = self.tanh_3(conv3)
        conv3 = conv3.permute(0, 2, 1) # B x R x C"
        if self.residual:
            conv3 = conv3 + identity
        return conv3


class SuperPixels(object):
    def __init__(self, h, w, f):
        self.update(h, w, f)
        self.pixels = []
        self.c_mean = None
        self.c_std = None
        self.c_max = None
        self.c_min = None

        self.p_mean = None
        self.p_std = None
        self.p_max = None
        self.p_min = None

    def update(self, h, w, f):
        self.h = h
        self.w = w
        self.f = f
        
class SLICPyTorch(nn.Module):
    def __init__(self, num_seg, m, num_iter):
        super().__init__()
        self.k = num_seg
        self.m = m
        self.num_iter = num_iter
    
    def make_superPixel(self, h, w, img):
        return SuperPixels(h, w, img[h, w])

    def initial_cluster_center(self, S, img, img_h, img_w, clusters):
        h = S // 2
        w = S // 2
        while h < img_h:
            while w < img_w:
                clusters.append(self.make_superPixel(h, w, img))
                w += S
            w = S // 2
            h += S
        return clusters

    def calc_gradient(self, h, w, img, img_w, img_h):
        if w + 1 >= img_w:
            w = img_w - 2 
        if h + 1 >= img_h:
            h = img_h - 2 
        grad = torch.sum(img[h + 1, w+1] - img[h, w])
        return grad

    def reassign_cluster_center_acc_to_grad(self, clusters, img, img_w, img_h):
        for c in clusters:
            cluster_gradient = self.calc_gradient(c.h, c.w, img, img_w, img_h)
            for dh in range(-1, 2):
                for dw in range(-1, 2):
                    H = c.h + dh
                    W = c.w + dw
                    new_gradient = self.calc_gradient(H, W, img, img_w, img_h)
                    if new_gradient < cluster_gradient:
                        c.update(H, W, img[H, W])
                        cluster_gradient = new_gradient

    def assign_pixels_to_cluster(self, clusters, S, img, img_h, img_w, tag, dis):
        for c in clusters:
            for h in range(c.h -2 * S, c.h + 2*S):
                if h < 0 or h >= img_h:
                    continue
                for w in range(c.w -2*S, c.w+2*S):
                    if w < 0 or w >= img_w:
                        continue
                    features = img[h, w]
                    Dc = torch.sqrt(torch.sum(torch.pow(features - c.f, 2)))
                    Ds = torch.sqrt(torch.pow(torch.tensor(h-c.h), 2)+torch.pow(torch.tensor(w-c.w), 2))
                    D = torch.sqrt(torch.pow(Dc / self.m, 2) + torch.pow(Ds /S, 2))
                    if D < dis[h, w]:
                        if (h, w) not in tag:
                            tag[(h, w)] = c
                            c.pixels.append([h ,w])
                        else:
                            tag[(h, w)].pixels.remove([h, w])
                            tag[(h, w)] = c
                            c.pixels.append([h, w])
                        dis[h, w] = D
    
    def update_cluster_mean(self, clusters, img):
        for c in clusters:
            coords_y = torch.tensor(c.pixels)[:, 0]
            coords_x = torch.tensor(c.pixels)[:, 1]
            H = torch.mean(coords_y.float()).int()
            W = torch.mean(coords_x.float()).int()

            c.c_mean = torch.mean(img[coords_y, coords_x], dim=0)
            c.c_std = torch.std(img[coords_y, coords_x], dim=0)
            c.c_max = torch.max(img[coords_y, coords_x], dim=0)
            c.c_min = torch.min(img[coords_y, coords_x], dim=0)
            c.update(H, W, img[H, W])

    def forward(self, imgs):
        b, img_h, img_w, f = imgs.size()
        N = img_h*img_w
        S = int(math.sqrt(N/self.k))
        all_clusters = []
        for img in imgs:
            clusters = []
            tag = {}
            dis = torch.full((img_h, img_w), torch.inf)
            clusters = self.initial_cluster_center(S, img, img_h, img_w, clusters)
            self.reassign_cluster_center_acc_to_grad(clusters, img, img_w, img_h)
            for i in range(self.num_iter):
                self.assign_pixels_to_cluster(clusters, S, img, img_h, img_w, tag, dis)
                self.update_cluster_mean(clusters, img)
            all_clusters.append(clusters)
        return all_clusters















