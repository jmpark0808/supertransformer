import numpy as np
import os
import sys
sys.path.insert(0, '/home/eddie/waterloo/supertransformer')
from dataset.superpixel_fast import SPDataset as SPFDataset
from dataset.superpixel_pyg import SPDataset as SPGDataset
from torch_geometric.loader import DataLoader as GDL
from torch.utils.data import DataLoader as DL
import torch
import torch.nn as nn
from torch_geometric.nn.dense.linear import Linear
from Models.SP_TFM_PyG import SP_TFM_PyG
from Models.SP_TFM import SP_TFM_REL
import time
from torch_geometric.loader import NeighborLoader



train_dir = '/mnt/hdd/Datasets/DUTS/TR/'


image_list = np.array(sorted([os.path.join('{}/Image'.format(train_dir), f) for f in os.listdir('{}/Image'.format(train_dir))]))
mask_list = np.array(sorted([os.path.join('{}/Mask'.format(train_dir), f) for f in os.listdir('{}/Mask'.format(train_dir))]))

image_list = image_list[:int(len(image_list)*0.01)]
mask_list = mask_list[:int(len(mask_list)*0.01)]


spg_dataset = SPGDataset(image_list, mask_list, 625, 256, 10, 'SPGFFT', True, 10, False, False, None, None, 7)

spg_loader = GDL(spg_dataset, 3, False, num_workers=4)

loader = NeighborLoader(
    spg_dataset,
    num_neighbors=[20, 15, 10],
    batch_size=128,
)
batch = next(iter(loader))
print(batch.x.size())