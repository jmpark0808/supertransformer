
import numpy as np
import os
import sys
from tqdm import tqdm
import time
sys.path.insert(0, '/home/eddie/waterloo/supertransformer')

from dataset.superpixel_pyg import SPDataset, SPDatasetExport
from torch_geometric.loader import DataLoader
test_dir = '/mnt/dragon/Datasets/DUTS/DUTS-TE'
image_list = np.array(sorted([os.path.join(os.path.join(test_dir, 'Image'), f) for f in os.listdir(os.path.join(test_dir, 'Image'))]))
mask_list = np.array(sorted([os.path.join(os.path.join(test_dir, 'Mask'), f) for f in os.listdir(os.path.join(test_dir, 'Mask'))]))


dummy_test = SPDatasetExport(image_list, mask_list, 625,
                               300, 10, 'SPGFFT',False, 
                               10, False, False, 0, 1, 0)
dummy_test_loader = DataLoader(
                dummy_test, batch_size=16, 
                num_workers=4, pin_memory=False)
for batch in tqdm(dummy_test_loader):
    pass



dataset = SPDataset(image_list, mask_list, 625, 300, 10, 'SPGFFT', True, 10, False, False, 0, 1, 0)
loader = DataLoader(dataset, batch_size=128, shuffle=True, pin_memory=False, num_workers=4)

start = time.time()
for ind, batch in enumerate(loader):
    print(time.time()-start)
    start = time.time()
    