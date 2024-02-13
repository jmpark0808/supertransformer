
import numpy as np
import os
import sys
sys.path.insert(0, '/home/eddie/waterloo/supertransformer')

from dataset.superpixel_pyg import SPDataset as SPGDataset
from dataset.superpixel_pyg import SPDatasetExport as SPGDatasetExport
from torch_geometric.loader import DataLoader as GDL
from torch.utils.data import DataLoader as DL
import torch
import torch.nn as nn
from torch_geometric.nn.dense.linear import Linear
from Models.SP_TFM_PyG import SP_TFM_PyG
from Models.SP_TFM import SP_TFM_REL
import time
from Blocks.CustomGUNet import GraphUNet




train_dir = '/mnt/hdd/Datasets/DUTS/TR/'


image_list = np.array(sorted([os.path.join('{}/Image'.format(train_dir), f) for f in os.listdir('{}/Image'.format(train_dir))]))
mask_list = np.array(sorted([os.path.join('{}/Mask'.format(train_dir), f) for f in os.listdir('{}/Mask'.format(train_dir))]))

image_list = image_list[:int(len(image_list)*0.01)]
mask_list = mask_list[:int(len(mask_list)*0.01)]

dummy_tr = SPGDatasetExport(image_list, mask_list, 625, 256, 10, 'SPGFFT', True, 10, False, False, None, 1)
dummy_tr_loader = GDL(
        dummy_tr, batch_size=16, 
        num_workers=4, shuffle=False, pin_memory=False)

for batch in dummy_tr_loader:
    pass


spg_dataset = SPGDataset(image_list, mask_list, 625, 256, 10, 'SPGFFT', True, 10, False, False, None, 1)

spg_loader = GDL(spg_dataset, 1, False, num_workers=4)
model = GraphUNet(36, 16, 1, 3, 0.5).cuda()

for batch in spg_loader:
    features = batch[0]
    seq_mask = batch[1]
    segments = batch[2]
    mask = batch[3]

    # np.save(f'/mnt/dragon/gat_logs/features_x_{batch_idx}', features.x.detach().cpu().numpy())
    # np.save(f'/mnt/dragon/gat_logs/features_ei_{batch_idx}', features.edge_index.detach().cpu().numpy())
    # np.save(f'/mnt/dragon/gat_logs/seq_mask_{batch_idx}', seq_mask.detach().cpu().numpy())
    # torch.save(self.model.state_dict(), f'/mnt/dragon/gat_logs/model_weight_{batch_idx}.pt')
    features = features.cuda()
    mask = mask.cuda()

    model(features.x[:, 2:], features.edge_index)
    assert(0)
