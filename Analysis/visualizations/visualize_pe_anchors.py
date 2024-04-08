import numpy as np
import os
import sys
sys.path.insert(0, '/home/eddie/waterloo/supertransformer')
from dataset.superpixel_fast import SPDataset as SPFDataset
from torch_geometric.loader import DataLoader as GDL
from torch.utils.data import DataLoader as DL
import torch
import torch.nn as nn
from torch_geometric.nn.dense.linear import Linear
from Models.SP_SWIN import SP_SWIN
import time
import torch_geometric
from PIL import Image
from skimage import segmentation
import matplotlib.pyplot as plt
import matplotlib

torch.manual_seed(22)
random_vector = torch.randn(1, 64).repeat(100, 1)
xs = torch.arange(5, 100, 10)
ys = torch.arange(5, 100, 10)

c = []
for x in xs:
    for y in ys:
        c.append([x, y])

c = torch.tensor(c)
distances = torch.sqrt(torch.sum(torch.pow(c.unsqueeze(1) - c.unsqueeze(0), 2), dim=-1))


cos = nn.CosineSimilarity(dim=0)
output = distances.reshape(10, 10, -1)
import matplotlib.pyplot as plt
import numpy as np
fig, ax = plt.subplots(10, 10)
count = 0 
for k in range(10):
    for l in range(10):
        patches = []
        for i in range(10):
            for j in range(10):
                patches.append(cos(output[k, l], output[i, j]).detach().cpu().numpy())

        # plt.imshow(np.array(patches).reshape(32, 32), cmap='hot')
        # plt.scatter(l, k, c='green', marker='s')
        # plt.title(f'Seed row {k}, column {l}')
        # plt.savefig(f"/home/eddie/Downloads/gif_pe/{count}.png")
        # plt.clf()
        # count += 1
        ax[k, l].imshow(np.array(patches).reshape(10, 10), cmap='hot')
        ax[k, l].set_xticks([])
        ax[k, l].set_yticks([])
        if l == 0:
            ax[k, l].set_ylabel(f'{k+1}') 
        if k == 9:
            ax[k, l].set_xlabel(f'{l+1}')
    
fig.suptitle('SuperFormer Positional Encoding Cosine Similarity')
plt.show()


assert(0)


