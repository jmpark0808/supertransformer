import numpy as np
import os
import sys
sys.path.insert(0, '/home/eddie/waterloo/supertransformer')
# from dataset.superpixel_fast import SPDataset as SPFDataset
from dataset.superpixel import SPDataset
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

train_dir = '/home/eddie/Datasets/DUTS/DUTS-TR/'


image_list = np.array(sorted([os.path.join('{}/Image'.format(train_dir), f) for f in os.listdir('{}/Image'.format(train_dir))]))
mask_list = np.array(sorted([os.path.join('{}/Mask'.format(train_dir), f) for f in os.listdir('{}/Mask'.format(train_dir))]))

image_list = image_list[:int(len(image_list)*0.01)]
mask_list = mask_list[:int(len(mask_list)*0.01)]


spg_dataset = SPDataset(image_list, mask_list, 1024, 320, 10, False, 'SPFFT',10, False)

spg_loader = DL(spg_dataset, 1, False, num_workers=4)


# state_dict = torch.load('/home/eddie/1105-220615_52378818/epoch=296-step=1486188.ckpt')

# for key in list(state_dict['state_dict'].keys()):
#     state_dict['state_dict'][key.replace('supert.', '')] = state_dict['state_dict'].pop(key)

# print(state_dict['state_dict'].keys())
dummy_linear = nn.Linear(2, 64).cuda()
# pyg = SP_SWIN(36, 32, 64, 8, 6, 0, 0, [4, 4, 4], 4).cuda()
# dummy_linear.weight = nn.Parameter(state_dict['state_dict']['locations.0.weight'])
# dummy_linear.bias = nn.Parameter(state_dict['state_dict']['locations.0.bias'])
dummy_linear.eval()



for batch in spg_loader:
    
 

    features = batch['features']
    img_name = batch['file_name']
    # img = Image.open(img_name[0]).resize((224, 224))
    # plt.imshow(img)
    # for i in range(0, 224, 32):
    #     if i != 224 and i !=0:
    #         plt.axhline(y=i, c='r', linewidth=3)
    #         plt.axvline(x=i, c='r', linewidth=3)
    # plt.show()
    

    # np.save(f'/mnt/dragon/gat_logs/features_x_{batch_idx}', features.x.detach().cpu().numpy())
    # np.save(f'/mnt/dragon/gat_logs/features_ei_{batch_idx}', features.edge_index.detach().cpu().numpy())
    # np.save(f'/mnt/dragon/gat_logs/seq_mask_{batch_idx}', seq_mask.detach().cpu().numpy())
    # torch.save(self.model.state_dict(), f'/mnt/dragon/gat_logs/model_weight_{batch_idx}.pt')
    features = features.cuda()
    

    centroids = features[:, :, :2]
    centroids_h = features[:, :, None, :2]

    
    desired_indices = torch.arange(0, centroids_h.size(1), 16)

    centroids_w = features[:, None, :, :2]
    relative_centroids = torch.norm(centroids_h - centroids_w, p=2, dim=-1)
    
  
    output = dummy_linear(centroids)[0]
    # plt.rcParams['axes.facecolor']='black'
    # for i in range(32):
    #     if i % 2 == 0:
    #         plt.scatter(centroids[0, i*32:i*32+32, 1].detach().cpu().numpy(), -centroids[0, i*32:i*32+32, 0].detach().cpu().numpy(), c=list(range(31, -1, -1)), cmap='YlOrBr')
    #     else:
    #         plt.scatter(centroids[0, i*32:i*32+32, 1].detach().cpu().numpy(), -centroids[0, i*32:i*32+32, 0].detach().cpu().numpy(), c=list(range(32)), cmap='GnBu')
    
    # plt.show()
    cos = nn.CosineSimilarity(dim=0)
    layernorm = nn.LayerNorm([64]).cuda()
    output = layernorm(output.reshape(32, 32, -1))
    relative_centroids_reshape_tensor = relative_centroids.reshape(32, 32, -1)
    
    centroids_reshape_tensor = centroids.reshape(32, 32, -1)
    import matplotlib.pyplot as plt
    import numpy as np
    # fig, ax = plt.subplots(32, 32)
    count = 0 
    l = 16
    k = 16
    patches = []
    for i in range(32):
        for j in range(32):
            patches.append(cos(relative_centroids_reshape_tensor[k, l], relative_centroids_reshape_tensor[i, j]).detach().cpu().numpy())
            # patches.append(torch.sqrt(torch.sum(torch.pow(output[k, l]-output[i, j], 2))).detach().cpu().numpy())
            # patches.append(torch.sqrt(torch.sum(torch.pow(relative_centroids_reshape_tensor[k, l]-relative_centroids_reshape_tensor[i, j], 2))).detach().cpu().numpy())
            

    # plt.imshow(np.array(patches).reshape(32, 32), cmap='hot')
    centroids_reshape = centroids.reshape(32, 32, -1).detach().cpu().numpy()
    plt.figure(figsize=(5,5))
    plt.scatter(centroids[0, :, 1].detach().cpu().numpy(), -centroids[0, :, 0].detach().cpu().numpy(), c=patches, cmap='jet')
    plt.scatter(centroids_reshape[k, l, 1], -centroids_reshape[k, l, 0], c='red', marker='*', s=100)
    plt.title(f'Seed row {k}, column {l}')
    plt.axis('off')
    plt.show()
    plt.clf()
    count += 1
    #         ax[k, l].imshow(np.array(patches).reshape(32, 32), cmap='hot')
    #         ax[k, l].set_xticks([])
    #         ax[k, l].set_yticks([])
    #         if l == 0:
    #             ax[k, l].set_ylabel(f'{k+1}') 
    #         if k == 31:
    #             ax[k, l].set_xlabel(f'{l+1}')
        
    # fig.suptitle('SuperFormer Positional Encoding Euclidean distance')
    # plt.show()

    
    assert(0)


