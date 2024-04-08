import torch
import matplotlib.pyplot as plt
import numpy as np

a = torch.randn([1, 16, 32, 32])
unfold = torch.nn.Unfold(4, 8)
b = unfold(a)
print(b.size())
fold = torch.nn.Fold(32, 4, 8)
c = fold(b)
print(torch.equal(a, c))
# assert(0)

def scattered_partition(x, window_size, unfold):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    
    B, H, W, C = x.shape
    x = x.permute(0, 3, 1, 2) # B C H W
    x = unfold(x) # B C*4*4 64
    x = x.reshape(B, C, window_size, window_size, -1) # B, C, 4, 4, 64
    windows = x.permute(0, 4, 2, 3, 1).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

H, W = 4, 4
window_size = 2
shift_size = 1
img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
h_slices = (slice(0, -window_size),
            slice(-window_size, -shift_size),
            slice(-shift_size, None))
w_slices = (slice(0, -window_size),
            slice(-window_size, -shift_size),
            slice(-shift_size, None))
cnt = 0
for h in h_slices:
    for w in w_slices:
        img_mask[:, h, w, :] = cnt
        cnt += 1
print(img_mask)
unfold = torch.nn.Unfold(2, 2)
mask_windows =scattered_partition(img_mask, window_size, unfold)  # nW, window_size, window_size, 1
# mask_windows = window_partition(img_mask, window_size)
mask_windows = mask_windows.view(-1, window_size * window_size)
print(mask_windows)
attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
print(attn_mask.size())
attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
print(attn_mask)







# fig, ax = plt.subplots()
# for i in range(32):
#     for j in range(32):
#         c = int(img_mask[0, i, j, 0].detach().cpu().numpy())
#         ax.text(i+0.5, -j-0.5, str(c), va='center', ha='center')

# min_val = 0
# max_val = 32

# ax.set_xlim(min_val, max_val)
# ax.set_ylim(min_val, -max_val)
# ax.set_xticks(np.arange(max_val))
# ax.set_yticks(-np.arange(max_val))
# ax.grid()
# plt.show()