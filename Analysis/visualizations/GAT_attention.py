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
from Models.SP_GAT_PyG import SP_GAT_PyG
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


spg_dataset = SPGDataset(image_list, mask_list, 625, 300, 10, 'SPGFFT', True, 10, False, False, None, None, 7)

spg_loader = GDL(spg_dataset, 1, False, num_workers=4)


state_dict = torch.load('/home/eddie/Downloads/epoch=571-step=1282996.ckpt')

for key in list(state_dict['state_dict'].keys()):
    state_dict['state_dict'][key.replace('model.', '')] = state_dict['state_dict'].pop(key)

pyg = SP_GAT_PyG(26, 16, None, 0, 8, 6, 625).cuda()
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

        pred, att = pyg(features, True)
        # pred_numpy = torch.sigmoid(pred).detach().cpu().numpy() # batch, seq_len, 1

        # batch_size = mask.size(0)
        # img_size = mask.size(2)
        # segments_ = segments.reshape([batch_size, -1]) # batch, img_size^2

        
        # plt_image = pred_numpy[segments_[0]-1].reshape([img_size, img_size])
         
        # plt.imshow(plt_image, cmap='gray')
        # plt.show()
        # assert(0)

        adj_layer = {}
        for i in range(6):
            attention_scores = att[i]

            adjs = []
            for j in range(attention_scores.size(1)):
                adj = torch_geometric.utils.to_scipy_sparse_matrix(features.edge_index, attention_scores[:, j])
                adj = torch.tensor(adj.todense())
                
                adj[adj == 0 ] = -1000000000000000000
                adj = torch.softmax(adj, -1).detach().numpy()
                # adj = adj/np.max(adj, axis=-1)
                adjs.append(adj)


            adj_layer[i] = adjs
       
        
        
        for i in range(np.max(segments)+1):
            fig, ax = plt.subplots( 6, 8, figsize=(40, 30),num=1, clear=True)
            for j in range(6):
                for k in range(8):
                    out_ = np.copy(img)
                    out_[segments == (i+1)] = [0, 255, 0]

                    for neighbor in np.argwhere(dense_adj[i, :] == 1):
                        if neighbor != i:
                            
                            ind = np.argwhere(segmentation.find_boundaries(segments-1 == neighbor, mode='inner'))
                     
                            cmap = matplotlib.cm.get_cmap('plasma')
                            rgb = cmap(adj_layer[j][k][i][neighbor]*255)
                            out_[ind[:, 0], ind[:, 1]] = rgb[0][:3]*255
                    ax[j, k].imshow(out_)
                    if j == 0 :
                        ax[j, k].set_title(f'Head {k}')
                    if k == 0:
                        ax[j, k].set_ylabel(f'Layer {j}')
                    ax[j, k].set_xticks([])
                    ax[j, k].set_yticks([])


            plt.tight_layout()
            plt.subplots_adjust(wspace=0, hspace=0)
            fig.savefig(f'/home/eddie/Downloads/gif/{i}')
            plt.clf()

        assert(0)


