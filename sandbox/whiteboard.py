import numpy as np
import os
from dataset.superpixel_fast import SPDataset
from torch.utils.data import DataLoader
train_dir = '/home/eddie/Datasets/DUTS/DUTS-TR/'


image_list = np.array(sorted([os.path.join(os.path.join(train_dir, 'Image'), f) for f in os.listdir(os.path.join(train_dir, 'Image'))]))
mask_list = np.array(sorted([os.path.join(os.path.join(train_dir, 'Mask'), f) for f in os.listdir(os.path.join(train_dir, 'Mask'))]))

           

        

data_train = SPDataset(image_list, mask_list, 3136,
                        448, 'SPFFFT', True,
                            10, 4)
loader = DataLoader(
        data_train, batch_size=32, 
        num_workers=20, shuffle=True, pin_memory=True, drop_last=True)

for batch in loader:
    pass