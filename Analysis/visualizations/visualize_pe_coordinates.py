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

train_dir = '/mnt/dragon/Datasets/DUTS/DUTS-TR/'


image_list = np.array(sorted([os.path.join('{}/Image'.format(train_dir), f) for f in os.listdir('{}/Image'.format(train_dir))]))
mask_list = np.array(sorted([os.path.join('{}/Mask'.format(train_dir), f) for f in os.listdir('{}/Mask'.format(train_dir))]))

image_list = image_list[:int(len(image_list)*0.01)]
mask_list = mask_list[:int(len(mask_list)*0.01)]


spg_dataset = SPFDataset(image_list, mask_list, 1024, 320, 10, 'SPFFFT', True, 10, False, False, None, 7, False)

spg_loader = DL(spg_dataset, 1, False, num_workers=4)


state_dict = torch.load('/mnt/hdd/Experiments/garbage/models/state_dict/0331-110115_46281259/epoch=556-step=312477.ckpt')

for key in list(state_dict['state_dict'].keys()):
    state_dict['state_dict'][key.replace('supert.', '')] = state_dict['state_dict'].pop(key)


dummy_linear = nn.Linear(2, 64).cuda()
# pyg = SP_SWIN(36, 32, 64, 8, 6, 0, 0, [4, 4, 4], 4).cuda()
dummy_linear.weight = nn.Parameter(state_dict['state_dict']['pos_linear.weight'])
dummy_linear.bias = nn.Parameter(state_dict['state_dict']['pos_linear.bias'])
dummy_linear.eval()



for batch in spg_loader:
    
 

    features = batch['features']
    img_name = batch['file_name']

    
    features = features.cuda()
    

    centroids = features[:, :, :2]
  
 
    cos = nn.CosineSimilarity(dim=0)
    output = centroids[0].reshape(32, 32, -1)-160
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
        
    # fig.suptitle('Coordinates Cosine Similarity')
    # plt.show()

    
    assert(0)


