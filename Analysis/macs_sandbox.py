import torch
from ptflops import get_model_complexity_info

import sys
sys.path.insert(0, '/home/eddie/waterloo/supertransformer')
# from Blocks.GraphBlocks import *
from Models.SP_TFM import SP_TFM_FFT
import torch.nn as nn
import torch.functional as F
from fvcore.nn import FlopCountAnalysis
from fvcore.nn import flop_count_table
class Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.l1 = nn.TransformerEncoderLayer(16*8, 8, 16*8, 0, batch_first=True)
        # self.l1 = nn.TransformerEncoder(self.l1, 6)


    def forward(self, x):
        y = self.l1(x)
        return y 
    

class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout, concat=True, alpha=0.2):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.dropout = nn.Dropout(dropout)
        # self.matmul = torch.matmul
        self.elu = nn.ELU()

    def forward(self, x):
        h, adj = x
        Wh = torch.matmul(h, self.W) # h.shape: (B, N, in_features), Wh.shape: (B, N, out_features)
        e = self._prepare_attentional_mechanism_input(Wh)
        return e

        # zero_vec = -9e15*torch.ones_like(e)
        # attention = torch.where(adj > 0, e, zero_vec)
        # attention = torch.softmax(attention, dim=-1)
        # attention = self.dropout(attention)
        # h_prime = torch.matmul(attention, Wh)

        # if self.concat:
        #     return self.elu(h_prime)
        # else:
        #     return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (B, N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (B, N, 1)
        # e.shape (B, N, N)
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # broadcast add
        e = Wh1 + Wh2.permute(0, 2, 1)
        return self.leakyrelu(Wh1)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

model = GraphAttentionLayer(144, 16*8, 0, True,0.2)
# model = SP_TFM_FFT(146, 16, 8, 6, 0)
# model = Model()
adj = torch.ones([1,625, 625])
inp = torch.ones([1,625, 146])
# macs, params = profile(model, inputs=(inp, adj))
# macs, params = clever_format([macs, params], "%.3f")
# print(macs, params)

def prepare_input(resolution):
    x1 = torch.FloatTensor(1, 625, 144)
    x2 = torch.FloatTensor(1, 625, 625)
    return dict(x = [x1, x2])

macs, params = get_model_complexity_info(model, input_res=(1, 625, 144), input_constructor=prepare_input, as_strings=True,
                                           print_per_layer_stat=False, verbose=True)
print(macs, params)

class PruneTokens(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        


    def forward(self, x):
        y = torch.matmul(x, x.permute(0, 1, 3, 2))
        return y 
    
model = PruneTokens()
dummy_input = torch.randn(1, 8, 625, 128)
macs, params = get_model_complexity_info(model, input_res=(8, 625, 128), as_strings=True,
                                           print_per_layer_stat=False, verbose=True)
print(macs, params)

flops = FlopCountAnalysis(model, dummy_input)
print(flop_count_table(flops))


model = SP_TFM_FFT(146, 16, 8, 6, 0)
flops = FlopCountAnalysis(model, inp)
print(flop_count_table(flops))