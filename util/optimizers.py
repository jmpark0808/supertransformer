
import torch
import torch.nn as nn
import torch.functional as F

class SoftTargetCrossEntropy(nn.Module):

    def __init__(self):
        super(SoftTargetCrossEntropy, self).__init__()
        self.softmax = torch.nn.LogSoftmax(dim=-1)

    def forward(self, x, target):
        loss = torch.sum(-target * self.softmax(x), dim=-1)
        return loss.mean()
