import torch

node_index = torch.arange(1024).reshape(1, 32, 32).float().cuda()


window_size = 4
local_unfold = torch.nn.Unfold(window_size, stride=window_size)
dilated_unfold = torch.nn.Unfold(window_size, dilation=(32//window_size))


local_index = local_unfold(node_index)
shifted_index = local_unfold(torch.roll(node_index, shifts=(-window_size//2, -window_size//2), dims=(0, 1)))
dilated_index = dilated_unfold(node_index)

all_local_indices = []
all_shifted_indices = []
all_dilated_indices = []
for i in range(local_index.shape[1]):
    x, y = torch.meshgrid(local_index[:, i], local_index[:, i])
    all_local_indices.append(torch.stack((x, y), dim=0).reshape(2, -1))
    x, y = torch.meshgrid(shifted_index[:, i], shifted_index[:, i])
    all_shifted_indices.append(torch.stack((x, y), dim=0).reshape(2, -1))
    x, y = torch.meshgrid(dilated_index[:, i], dilated_index[:, i])
    all_dilated_indices.append(torch.stack((x, y), dim=0).reshape(2, -1))
    
all_local_indices = torch.cat(all_local_indices, dim=1)
all_shifted_indices = torch.cat(all_shifted_indices, dim=1)
all_dilated_indices = torch.cat(all_dilated_indices, dim=1)

print(all_local_indices.size())
