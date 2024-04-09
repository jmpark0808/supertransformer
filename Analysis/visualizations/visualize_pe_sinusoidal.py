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
import math

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

def positionalencoding2d(d_model, height, width):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    pe = torch.zeros(d_model, height, width)
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0., d_model, 2) *
                         -(math.log(10000.0) / d_model))
    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_h = torch.arange(0., height).unsqueeze(1)
    pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

    return pe




cos = nn.CosineSimilarity(dim=0)
output = positionalencoding2d(64, 32, 32).permute(1,2, 0)

import matplotlib.pyplot as plt
import numpy as np
# fig, ax = plt.subplots(32, 32)
count = 0 
for k in range(32):
    for l in range(32):
        patches = []
        for i in range(32):
            for j in range(32):
                patches.append(cos(output[k, l], output[i, j]).detach().cpu().numpy())

        plt.imshow(np.array(patches).reshape(32, 32), cmap='hot')
        plt.scatter(l, k, c='green', marker='s')
        plt.title(f'Seed row {k}, column {l}')
        plt.savefig(f"/home/eddie/Downloads/gif_pe/{count}.png")
        plt.clf()
        count += 1
#         ax[k, l].imshow(np.array(patches).reshape(32, 32), cmap='hot')
#         ax[k, l].set_xticks([])
#         ax[k, l].set_yticks([])
#         if l == 0:
#             ax[k, l].set_ylabel(f'{k+1}') 
#         if k == 31:
#             ax[k, l].set_xlabel(f'{l+1}')
    
# fig.suptitle('Sinusoidal Positional Encoding Cosine Similarity')
# plt.show()


assert(0)


