import torch
import torch.nn as nn
print(f"gpu used {torch.cuda.max_memory_allocated()} memory")
# gpu used 0 memory


centroids = torch.randn(100, 56, 56, 2).cuda()
locations = nn.Linear(3136, 32).cuda()
centroids_h = centroids.reshape(centroids.size(0), -1, centroids.size(3))[:, :, None, :]
centroids_w = centroids.reshape(centroids.size(0), -1, centroids.size(3))[:, None, :, :]

centroids = torch.sqrt(torch.sum(torch.pow(centroids_h - centroids_w, 2), -1)).triu(diagonal=1) 
# centroids = centroids.reshape(centroids.size(0),
#                                                     int(centroids.size(1)**0.5),
#                                                     int(centroids.size(1)**0.5) , -1)


# centroids = locations(centroids)
# centroids = centroids.reshape(centroids.size(0), -1, centroids.size(3))
print(f"gpu used {torch.cuda.max_memory_allocated()/(1024**2)} memory")
# gpu used 8192 memory

# # should return same memory usage as nothing was deleted
# torch.cuda.reset_peak_memory_stats(device=None)
# print(f"gpu used {torch.cuda.max_memory_allocated(device=None)} memory")
# # gpu used 8192 memory

# # delete tensors and reduce peak memory usage
# del y
# torch.cuda.reset_peak_memory_stats(device=None)
# print(f"gpu used {torch.cuda.max_memory_allocated(device=None)} memory")
# # gpu used 4096 memory

# del x
# torch.cuda.reset_peak_memory_stats(device=None)
# print(f"gpu used {torch.cuda.max_memory_allocated(device=None)} memory")
# # gpu used 0 memory