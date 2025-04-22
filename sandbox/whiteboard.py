import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from torch_geometric.data import Data
import random

# Parameters
H, W = 64, 64
square_size = 16
num_superpixels = 100
samples = 10  # for visualization, normally use 10k
radius = 3

def generate_sample():
    # Create a blank image with a centered square
    image = np.zeros((H, W), dtype=np.float32)
    center = H // 2
    half = square_size // 2
    image[center - half:center + half, center - half:center + half] = 1.0

    # Simulate superpixels by generating random blob centroids
    points, _ = make_blobs(n_samples=num_superpixels, centers=1, cluster_std=15, center_box=(0, H))
    points = np.clip(points, 0, H - 1).astype(np.int32)

    # Build superpixel graph
    features = []
    positions = []
    labels = []

    for x, y in points:
        x = int(x)
        y = int(y)
        if 0 <= x < H and 0 <= y < W:
            intensity = image[x, y]
            features.append([intensity])
            positions.append([x / H, y / W])  # normalized static position
            labels.append(int(intensity > 0.5))  # label: 1 if in center square

    features = torch.tensor(features, dtype=torch.float32)
    positions = torch.tensor(positions, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.long)

    # Create edge index based on distance threshold
    edge_index = []
    N = len(positions)
    for i in range(N):
        for j in range(N):
            if i != j:
                dist = torch.norm(positions[i] - positions[j])
                if dist < radius / H:  # normalized radius
                    edge_index.append([i, j])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    # Pack into PyG Data object
    data = Data(x=features, edge_index=edge_index, pos=positions, y=labels)

    return data, image, points

# Generate a few samples for visualization
dataset = [generate_sample() for _ in range(samples)]


# Visualize one example
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
data, img, pts = dataset[0]

axes[0].imshow(img, cmap='gray')
axes[0].set_title("Input Image with Center Object")
axes[1].imshow(img, cmap='gray')
axes[1].scatter(pts[:, 1], pts[:, 0], c=data.y.numpy(), cmap='coolwarm', edgecolors='k')
axes[1].set_title("Superpixels Colored by Label")
plt.tight_layout()
plt.show()