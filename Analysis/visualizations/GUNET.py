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
from Models.SP_GUNET import SP_GUNET_PyG
import time
import torch_geometric
from PIL import Image
from skimage import segmentation
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.lines import Line2D

train_dir = '/mnt/dragon/Datasets/DUTS/DUTS-TR/'


image_list = np.array(sorted([os.path.join('{}/Image'.format(train_dir), f) for f in os.listdir('{}/Image'.format(train_dir))]))
mask_list = np.array(sorted([os.path.join('{}/Mask'.format(train_dir), f) for f in os.listdir('{}/Mask'.format(train_dir))]))

image_list = image_list[:int(len(image_list)*0.01)]
mask_list = mask_list[:int(len(mask_list)*0.01)]


spg_dataset = SPGDataset(image_list, mask_list, 625, 300, 10, 'SPGFFT', True, 10, False, False, None, 1)

spg_loader = GDL(spg_dataset, 1, False, num_workers=4)


state_dict = torch.load('/mnt/hdd/Experiments/garbage/models/state_dict/0214-090034_/epoch=174-step=49175.ckpt')

for key in list(state_dict['state_dict'].keys()):
    state_dict['state_dict'][key.replace('model.', '')] = state_dict['state_dict'].pop(key)

pyg = SP_GUNET_PyG(36, 16, 8, 625, 3, 0).cuda()
pyg.load_state_dict(state_dict['state_dict'])
pyg.eval()


for batch in spg_loader:
    
 

    features = batch[0]
    seq_mask = batch[1]
    segments = batch[2]
    mask = batch[3]
    img_name = batch[4]

    # np.save(f'/mnt/dragon/gat_logs/features_x_{batch_idx}', features.x.detach().cpu().numpy())
    # np.save(f'/mnt/dragon/gat_logs/features_ei_{batch_idx}', features.edge_index.detach().cpu().numpy())
    # np.save(f'/mnt/dragon/gat_logs/seq_mask_{batch_idx}', seq_mask.detach().cpu().numpy())
    # torch.save(self.model.state_dict(), f'/mnt/dragon/gat_logs/model_weight_{batch_idx}.pt')
    features = features.cuda()
    mask = mask.cuda()
    img = Image.open(img_name[0]).convert('RGB')
    img = img.resize((300, 300))
    img = np.array(img)
    dense_adj = np.squeeze(torch_geometric.utils.to_dense_adj(features.edge_index).cpu().detach().numpy())
    
    segments = np.squeeze(segments.cpu().detach().numpy())
    # forward pass
    with torch.no_grad():

        pred, perms, edge_indices = pyg(features, True)
        
        initial_perm = np.arange(625)
        fig, ax = plt.subplots(1, len(perms), num=1, clear=True)
        for idx, (perm, edge) in enumerate(zip(perms, edge_indices[1:])):
            out_ = np.copy(img)
            initial_perm = initial_perm[perm.detach().cpu().numpy()]
            
            for p in initial_perm:
                ind = np.argwhere(segmentation.find_boundaries(segments-1 == p, mode='inner'))
                out_[ind[:, 0], ind[:, 1]] = [0, 255, 0]

            centroids = features.x[:, :2]
            for e in edge.T:
                y0, x0 = centroids[initial_perm[e.detach().cpu().numpy()[0]]].detach().cpu().numpy()
                y1, x1 = centroids[initial_perm[e.detach().cpu().numpy()[1]]].detach().cpu().numpy()
                l = Line2D([x0,x1],[y0,y1], alpha=0.5)
                ax[idx].add_line(l)
            ax[idx].imshow(out_)
        plt.show()
        
        
        
        
        
            
           

