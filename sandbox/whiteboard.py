from typing import Callable, Optional, Union

import torch
from torch import Tensor

from torch_geometric.nn.inits import uniform
from torch_geometric.nn.pool.select import Select, SelectOutput
from torch_geometric.nn.resolver import activation_resolver
from torch_geometric.utils import cumsum, scatter, softmax




batch = torch.arange(0, 10).unsqueeze(-1).repeat(1, 5).view(-1)
print(batch)