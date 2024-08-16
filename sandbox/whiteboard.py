from typing import Callable, Optional, Union
from einops import rearrange
import torch
from torch import Tensor
import torch.nn.functional as F
import time
from Blocks.performer2 import ViP, ViPEnc
import numpy as np
from fvcore.nn import FlopCountAnalysis, flop_count_table, parameter_count
# model = ViP(image_size=32, patch_size=1, dim=32, depth=6, heads=2, mlp_dim=32*4, channels=16, dim_head=16).cuda()
# model.eval()




# dummy_input = torch.randn(1, 1024, 38).cuda()
# l = []
# with torch.no_grad():
#     for i in range(1000):
#         start = time.time()
#         model(dummy_input)
#         end = time.time()
#         l.append(end-start)

# print(np.mean(l))


# model = ViPEnc(image_size=32, patch_size=1, dims=[32, 64, 128, 256], depths=[2, 2, 6, 2], heads=[2,4, 8, 16], mlp_ratio=4, channels=16).cuda()
# model.eval()

# l = []
# with torch.no_grad():
#     for i in range(1000):
#         start = time.time()
#         model(dummy_input)
#         end = time.time()
#         l.append(end-start)

# print(np.mean(l))


linear = torch.nn.Linear(256, 256)
class conv(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv1d(256, 256*8, 1, groups=8)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = rearrange(x, 'b (c g) n->b c g n', g=8)
        x = x.mean(2)
        return x
    
conv_model = conv()

params_linear = parameter_count(linear)['']
params_conv = parameter_count(conv_model)['']

inp = torch.randn([1, 1024, 256])
flops_linear = FlopCountAnalysis(linear, inp)
flops_conv = FlopCountAnalysis(conv_model, inp)
flops_linear = flops_linear.total()
flops_conv = flops_conv.total()

print(params_linear, flops_linear)
print(params_conv, flops_conv)
