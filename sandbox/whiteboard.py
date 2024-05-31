import torch

a = torch.zeros(1, 4, 4, 1)
b = torch.ones(1, 4, 4, 1)
c = torch.ones(1, 4, 4, 1)*2
d = torch.ones(1, 4, 4, 1)*3


t = torch.cat([a, b ], 1)
h = torch.cat([c, d], 1)
f = torch.cat([t, h], 2)

# print(torch.squeeze(f))

k = torch.stack([a, b, c, d], 1)
print(k.reshape(2, 2, 4, 4).permute(2, 0, 3, 1).reshape(8, 8))