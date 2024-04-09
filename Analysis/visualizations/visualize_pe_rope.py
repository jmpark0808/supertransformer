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
from einops import rearrange, repeat

dim = 64
torch.manual_seed(2)
random_vectors = torch.randn(1, 1, dim)
random_vectors = random_vectors.repeat(1, 625, 1)
def rotate_every_two(x):
    x = rearrange(x, '... (d j) -> ... d j', j = 2)
    x1, x2 = x.unbind(dim = -1)
    x = torch.stack((-x2, x1), dim = -1)
    return rearrange(x, '... d j -> ... (d j)')

class AxialRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_freq = 10):
        super().__init__()
        self.dim = dim
        scales = torch.linspace(1., max_freq / 2, self.dim // 4)
        self.register_buffer('scales', scales)

    def forward(self, x):
        device, dtype, n = x.device, x.dtype, int(math.sqrt(x.shape[-2]))

        # seq = torch.linspace(-1., 1., steps = n, device = device)
        # seq = seq.unsqueeze(-1)

        # scales = self.scales[(*((None,) * (len(seq.shape) - 1)), Ellipsis)]
        # scales = scales.to(x)
        seq = torch.arange(1, n+1)
        seq = seq.unsqueeze(-1)

        d = torch.arange(1, self.dim//4+1)
        theta = (10000**(-2.*(d-1)/self.dim))
        theta = theta.unsqueeze(0)

        seq = seq * theta

        x_sinu = repeat(seq, 'i d -> i j d', j = n)
        y_sinu = repeat(seq, 'j d -> i j d', i = n)

        sin = torch.cat((x_sinu.sin(), y_sinu.sin()), dim = -1)
        cos = torch.cat((x_sinu.cos(), y_sinu.cos()), dim = -1)

        sin, cos = map(lambda t: rearrange(t, 'i j d -> (i j) d'), (sin, cos))
        sin, cos = map(lambda t: repeat(t, 'n d -> () n (d j)', j = 2), (sin, cos))
        return sin, cos


ax = AxialRotaryEmbedding(dim, 500)
sin, cos = ax(random_vectors)
q, k = map(lambda t: (t * cos) + (rotate_every_two(t) * sin), (random_vectors, random_vectors))


cos = nn.CosineSimilarity(dim=0)
output = q.reshape(25, 25, -1)
import matplotlib.pyplot as plt
import numpy as np
# fig, ax = plt.subplots(25, 25)
count = 0 
for k in range(25):
    for l in range(25):
        patches = []
        for i in range(25):
            for j in range(25):
                patches.append(np.clip(cos(output[k, l], output[i, j]).detach().cpu().numpy(), 0, 1))


        plt.imshow(np.array(patches).reshape(25, 25), cmap='hot')
        plt.scatter(l, k, c='green', marker='s')
        plt.title(f'Seed row {k}, column {l}')
        plt.savefig(f"/home/eddie/Downloads/gif_rope/{count}.png")
        plt.clf()
        count += 1
#         ax[k, l].imshow(np.array(patches).reshape(25, 25), cmap='hot', vmin=0, vmax=1)
#         ax[k, l].set_xticks([])
#         ax[k, l].set_yticks([])
#         if l == 0:
#             ax[k, l].set_ylabel(f'{k+1}') 
#         if k == 24:
#             ax[k, l].set_xlabel(f'{l+1}')
    
# fig.suptitle('Rotary Positional Encoding Cosine Similarity')
# plt.show()


assert(0)


