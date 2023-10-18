
import numpy as np
import os
import sys
sys.path.insert(0, '/home/eddie/waterloo/supertransformer')

from dataset.superpixel_pyg import SPDataset
from torch_geometric.loader import DataLoader
train_dir = '/home/eddie/Datasets/DUTS/DUTS-TR'


image_list = np.array(sorted([os.path.join(os.path.join(train_dir, 'Image'), f) for f in os.listdir(os.path.join(train_dir, 'Image'))]))
mask_list = np.array(sorted([os.path.join(os.path.join(train_dir, 'Mask'), f) for f in os.listdir(os.path.join(train_dir, 'Mask'))]))

dataset = SPDataset(image_list, mask_list, 625, 256, 10, 'SPGFFT', True, 10, False, False, None, None, 7)
loader = DataLoader(dataset, batch_size=128, shuffle=True, pin_memory=False, num_workers=9)

for ind, batch in enumerate(loader):
    print(ind)