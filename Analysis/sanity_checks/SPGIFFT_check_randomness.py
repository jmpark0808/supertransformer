import numpy as np
import os
import sys
sys.path.insert(0, '/mnt/pegasus/waterloo/supertransformer')
from dataset.superpixel_pyg_image import SPDataset as SPGIDataset
from dataset.superpixel_pyg import SPDataset as SPGDataset
from torch_geometric.loader import DataLoader as GDL
from torch.utils.data import DataLoader as DL
import torch
import torch.nn as nn
from torch_geometric.nn.dense.linear import Linear
from Models.SP_TFM_PyG import SP_TFM_PyG
from Models.SP_TFM import SP_TFM_REL
from Models.SP_GAT_PyG import SP_GAT_PyG
import time



train_dir = '/mnt/dragon/Datasets/DUTS/DUTS-TR/'


image_list = np.array(sorted([os.path.join('{}/Image'.format(train_dir), f) for f in os.listdir('{}/Image'.format(train_dir))]))
mask_list = np.array(sorted([os.path.join('{}/Mask'.format(train_dir), f) for f in os.listdir('{}/Mask'.format(train_dir))]))

image_list = image_list[:1]
mask_list = mask_list[:1]


spgi_dataset = SPGIDataset(image_list, mask_list, 625, 256, 10, 'SPGIFFT', True, 10, False, False, 7)


spf_loader = GDL(spgi_dataset, 1, False, num_workers=0)



# for a in spf_loader:
#     spgi_features_anchor = a[0]
#     spgi_seq_mask_anchor = a[1]
#     spgi_segments_anchor = a[2]
#     spgi_mask_anchor = a[3]


features_diff = []
seq_mask_diff = []
segments_diff = []
mask_diff = []

for _ in range(1000):
    for a in spf_loader:
        spgi_features = a[0]
        spgi_seq_mask = a[1]
        spgi_segments = a[2]
        spgi_mask = a[3]

        features_diff.append(spgi_features.x)
        seq_mask_diff.append(spgi_seq_mask)
        segments_diff.append(spgi_segments)
        mask_diff.append( spgi_mask)


features_diff = torch.stack(features_diff, dim=0)
seq_mask_diff = torch.stack(seq_mask_diff, dim=0)
segments_diff = torch.stack(segments_diff, dim=0)
mask_diff = torch.stack(mask_diff, dim=0)

features_max = torch.max(features_diff, dim=0)[0]
features_min = torch.min(features_diff, dim=0)[0]

seq_mask_max = torch.max(seq_mask_diff, dim=0)[0]
seq_mask_min = torch.min(seq_mask_diff, dim=0)[0]

segments_max = torch.max(segments_diff, dim=0)[0]
segments_min = torch.min(segments_diff, dim=0)[0]

mask_max = torch.max(mask_diff, dim=0)[0]
mask_min = torch.min(mask_diff, dim=0)[0]
torch.set_printoptions(precision=20)
torch.sum(torch.abs(features_max-features_min))
print(torch.sum(torch.abs(seq_mask_max - seq_mask_min)))
print(torch.sum(torch.abs(segments_max-segments_min)))
print(torch.sum(torch.abs(mask_max - mask_min)))
assert(0)

import matplotlib.pyplot as plt

fig, ax = plt.subplots(2, 2)
ax[0, 0].plot(features_diff)
ax[0, 1].plot(seq_mask_diff)
ax[1, 0].plot(segments_diff)
ax[1, 1].plot(mask_diff)

ax[0,0].set_xlabel('Sample idx')
ax[0,0].set_ylabel('Difference')
ax[0,0].set_title(f'Features diff sum {torch.sum(torch.tensor(features_diff))}')

ax[0,1].set_xlabel('Sample idx')
ax[0,1].set_ylabel('Difference')
ax[0,1].set_title(f'Seq mask diff sum {torch.sum(torch.tensor(seq_mask_diff))}')


ax[1,0].set_xlabel('Sample idx')
ax[1,0].set_ylabel('Difference')
ax[1,0].set_title(f'Segments diff sum {torch.sum(torch.tensor(segments_diff))}')


ax[1,1].set_xlabel('Sample idx')
ax[1,1].set_ylabel('Difference')
ax[1,1].set_title(f'Mask diff sum {torch.sum(torch.tensor(mask_diff))}')
plt.tight_layout()
plt.show()