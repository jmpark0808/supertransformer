import torch
groups = 2
a = torch.arange(1, 17)
b = a.reshape(2, 8)
print(b.permute(1, 0).reshape(-1))
