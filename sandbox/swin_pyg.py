import torch

node_index = torch.arange(1024).reshape(1, 32, 32).float()


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

a = torch.zeros([2, 16384])
b = torch.ones([2, 16384])
c = torch.ones([2, 16384])*2
edge_index = torch.cat((a, b, c, a, b, c), dim=1).long()
num_edges = (window_size**4)*((32//window_size)**2)

edge_index = edge_index.reshape(2, -1, 3, num_edges)
dilated_index = edge_index[:, :, 0, :].reshape(2, -1)
local_index = edge_index[:, :, 1, :].reshape(2, -1)
shifted_index = edge_index[:, :, 2, :].reshape(2, -1)
print(dilated_index)
print(local_index)
print(shifted_index)


# xs = torch.arange(0, 320, 10)
# ys = torch.arange(0, 320, 10)
# xs, ys = torch.meshgrid(xs, ys)
# xs = xs.reshape(-1)
# ys = ys.reshape(-1)
# import matplotlib.pyplot as plt
# plt.scatter(xs, -ys)
    


# for edge_ind in range(all_local_indices.size(1)):
#     plt.plot(xs[all_local_indices[:, edge_ind].long()].detach().cpu().numpy(), -ys[all_local_indices[:, edge_ind].long()].detach().cpu().numpy())
# plt.show()

# print(all_local_indices.size())
