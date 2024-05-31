import torch

x = torch.arange(64).reshape(1, 8, 8, 1)
B, H, W, C = x.shape
window_size = 4
x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
x = x.repeat_interleave(2, dim=1).repeat_interleave(2, dim=3)
windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
print(torch.squeeze(windows))