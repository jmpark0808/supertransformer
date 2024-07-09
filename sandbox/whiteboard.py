import torch

t = torch.arange(0, 1024).reshape(1, 32 ,32, 1)
B, H, W, C = t.shape
window_size = 8
x = t.view(1, window_size, H // window_size, window_size, W // window_size,  C)
    

windows = x.permute(0, 2, 4, 1, 3, 5).contiguous().view(-1, window_size, window_size, C)
    
B = int(windows.shape[0] / (H * W / window_size / window_size))
x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
x = x.permute(0, 3, 1, 4, 2, 5).contiguous().view(B, H, W, -1)
    
print(torch.sum(torch.abs(t-x)))
print(x)