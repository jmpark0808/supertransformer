import torch

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

def dilated_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B,  window_size, H // window_size, window_size, W // window_size, C)
    windows = x.permute(0, 2, 4, 1, 3, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
  
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

def dilation_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / ( H // window_size * W // window_size))
    x = windows.view(B,  H // window_size, W // window_size, window_size, window_size,  -1)
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous().view(B, H, W, -1)
    return x


test_input = torch.arange(1, 1025).reshape(1, 32, 32, 1)

output = dilated_partition(test_input, 4)
# print(output[0])
revert = dilation_reverse(output, 4, 32, 32)

print(torch.squeeze(revert))

# output = window_partition(test_input, 4)
# revert = window_reverse(output, 4, 32, 32)

# print(torch.squeeze(revert))
# for out in output:
#     print(revert.resize(64))

