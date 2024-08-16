from typing import Callable, Optional, Union

import torch
from torch import Tensor
import torch.nn.functional as F
import time
from Blocks.performer2 import ViP, ViPEnc
import numpy as np
model = ViP(image_size=32, patch_size=1, dim=32, depth=6, heads=2, mlp_dim=32*4, channels=16, dim_head=16).cuda()
model.eval()




dummy_input = torch.randn(1, 1024, 38).cuda()
l = []
with torch.no_grad():
    for i in range(1000):
        start = time.time()
        model(dummy_input)
        end = time.time()
        l.append(end-start)

print(np.mean(l))


model = ViPEnc(image_size=32, patch_size=1, dims=[32, 64, 128, 256], depths=[2, 2, 6, 2], heads=[2,4, 8, 16], mlp_ratio=4, channels=16).cuda()
model.eval()

l = []
with torch.no_grad():
    for i in range(1000):
        start = time.time()
        model(dummy_input)
        end = time.time()
        l.append(end-start)

print(np.mean(l))