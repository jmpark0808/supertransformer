from typing import Callable, Optional, Union

import torch
from torch import Tensor
import torch.nn.functional as F


input = torch.randn(1, 1024, 4, 64)

weights = torch.nn.Parameter(torch.randn(4, 64, 64))
bias = torch.nn.Parameter(torch.randn(4, 64))

output = torch.einsum("ijkl,klm->ijkm" ,input, weights)
output = output+bias
print(output.size())