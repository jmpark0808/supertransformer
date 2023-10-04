import numpy as np
import os
import sys
sys.path.insert(0, '/home/eddie/waterloo/supertransformer')
from dataset.superpixel_fast import SPDataset as SPFDataset
from dataset.superpixel_pyg import SPDataset as SPGDataset
from torch_geometric.loader import DataLoader as GDL
from torch.utils.data import DataLoader as DL
import torch

train_dir = '/mnt/hdd/Datasets/DUTS/DUTS-TR/'


image_list = np.array(sorted([os.path.join('{}/Image'.format(train_dir), f) for f in os.listdir('{}/Image'.format(train_dir))]))
mask_list = np.array(sorted([os.path.join('{}/Mask'.format(train_dir), f) for f in os.listdir('{}/Mask'.format(train_dir))]))

image_list = image_list[:int(len(image_list)*0.01)]
mask_list = mask_list[:int(len(mask_list)*0.01)]


spf_dataset = SPFDataset(image_list, mask_list, 625, 256, 10, 'SPF', True, 10, False, False, None, None, 7)
spg_dataset = SPGDataset(image_list, mask_list, 625, 256, 10, 'SPGFFT', True, 10, False, False, None, None, 7)

spf_loader = DL(spf_dataset, 1, False)
spg_loader = GDL(spg_dataset, 1, False)

for a, b in zip(spf_loader, spg_loader):
    a_features = a['features']
    a_seq_mask = a['seq_mask']
    a_segments = a['segments']
    a_mask = a['mask'].cpu()
    a_img = a['img']
    a_adj = a['neighbor_array']
    a_distances = a['edge_features']

    a_edge_index = np.array(np.nonzero(a_adj.detach().cpu().numpy()))

    b_features = b[0].x
    b_edge_index = b[0].edge_index
    b_edge_attr = b[0].edge_attr

    # print(torch.sum(torch.abs(a_features-b_features)))

    print(len(a_edge_index[0, :]), len(b_edge_index[0, :]))


