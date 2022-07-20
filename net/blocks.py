import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

def make_layers(cfg, in_channels):
    layers = []
    dilation_flag = False
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'm':
            layers += [nn.MaxPool2d(kernel_size=1, stride=1)]
            dilation_flag = True
        else:
            if not dilation_flag:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=2, dilation=2)
            layers += [conv2d, nn.ReLU()]
            in_channels = v
    return nn.Sequential(*layers)


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        configure = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'm', 512, 512, 512, 'm']
        self.seq = make_layers(configure, 3)


    def forward(self, x):
        conv1 = self.seq(x)

        return conv1

class EncoderNMP(nn.Module):
    def __init__(self):
        super(EncoderNMP, self).__init__()
        configure = [64, 64, 'm', 128, 128, 'm', 256, 256, 256, 'm', 256, 256, 256, 256, 256, 256]
        self.seq = make_layers(configure, 3)


    def forward(self, *input):
        x = input[0]
        conv1 = self.seq(x)
  

        return conv1