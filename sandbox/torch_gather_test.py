import torch

a = torch.randn(2, 2, 2, 2) # B, H, N, C
print(a)
b = torch.tensor([[0, 1], [1, 0]])
c = torch.gather(a, 2, b.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, 2))
print(c)
print(a[0, 0, 0, :], a[0, 1, 1, :], a[1, 0, 1, :], a[1, 1, 0, :])