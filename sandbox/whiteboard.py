import torch

x = torch.randn(10, 625)

y = torch.topk(x, 100)
print(y[0].size())