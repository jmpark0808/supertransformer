import torch
import torchsparse
import torchsparse.nn as spnn
from torchsparse import SparseTensor

# Step 1: Create a batch of 2 sparse 2D images
batch_size, H, W = 2, 5, 5
dense_batch = torch.zeros((batch_size, H, W), dtype=torch.float32)

# Add sparse values to each image
dense_batch[0, 1, 1] = 1.0
dense_batch[0, 2, 2] = 2.0
dense_batch[1, 3, 3] = 3.0
dense_batch[1, 4, 4] = 4.0

# Step 2: Extract non-zero coords and features for the whole batch
coords_list = []
features_list = []

for b in range(batch_size):
    coords = torch.nonzero(dense_batch[b], as_tuple=False)  # (Nᵢ, 2)
    batch_idx = torch.full((coords.shape[0], 1), b, dtype=torch.long)
    coords = torch.cat([batch_idx, coords], dim=1)  # (Nᵢ, 3)
    
    feats = dense_batch[b, coords[:, 1], coords[:, 2]].unsqueeze(1)
    coords_list.append(coords)
    features_list.append(feats)

coords = torch.cat(coords_list, dim=0)      # (N_total, 3)
features = torch.cat(features_list, dim=0)  # (N_total, 1)

# Step 3: Build SparseTensor
sparse_input = SparseTensor(coords=coords, feats=features)

# Step 4: Sparse convolution
conv = spnn.Conv2d(1, 4, kernel_size=3, stride=1)
sparse_output = conv(sparse_input)  # still a SparseTensor

# Step 5: Convert to dense (optional)
dense_output = sparse_output.to_dense(shape=(batch_size, 4, H, W))  # (B, C, H, W)
print("Output shape:", dense_output.shape)