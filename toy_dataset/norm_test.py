import numpy as np
import torch
import torch.nn as nn

random = torch.rand(1, 24, 2)
print(random)
norm = nn.LayerNorm(2)
random = norm(random)
print(random)
