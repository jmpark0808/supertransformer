import os
import matplotlib.pyplot as plt
import cv2
from skimage import io
from skimage.segmentation import mark_boundaries, slic
from skimage.measure import regionprops_table
from skimage import segmentation, color
import numpy as np
from PIL import Image
from tqdm import tqdm
import pickle

dataset_images = '/mnt/hdd/Datasets/DUTS/DUTS-TR/Image'
masks = '/mnt/hdd/Datasets/DUTS/DUTS-TR/Mask'
segment_numbers = [1024, 1024, 1024]
compactness = [10, 10, 10]
ec = [True, False, False]
sz = [True, True, False]
x_label = []
all_ious = []


from dataset.superpixel import SPDatasetDummy
from torch.utils.data import DataLoader
import torch
import os
from torch_geometric.utils import scatter
train_dir = '/mnt/dragon/Datasets/DUTS/DUTS-TR'
test_dir = '/mnt/dragon/Datasets/DUTS/DUTS-TE'
batch_size = 1
num_workers = 20
size = 320

dataloader = 'SP'

coeff = 10
ignore_phase = False

image_list = np.array(sorted([os.path.join('{}/Image'.format(train_dir), f) for f in os.listdir('{}/Image'.format(train_dir))]))[:]
mask_list = np.array(sorted([os.path.join('{}/Mask'.format(train_dir), f) for f in os.listdir('{}/Mask'.format(train_dir))]))[:]




tr_image_list = image_list
tr_mask_list = mask_list

# plt.figure(figsize=(10,10))
for seg, c, e, s in zip(segment_numbers, compactness, ec, sz):
    ious = []
    dataset = SPDatasetDummy(tr_image_list, tr_mask_list, seg, size, c, e, s)
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    for batch in tqdm(dl):
        
        
        num_seg = seg
        

        segments = batch['segments']
        msk = batch['mask'].detach().numpy()
        img = batch['features'].reshape(3,size, size).permute(1, 2, 0).detach().numpy()

        # segments_np = segments.squeeze().detach().numpy()
        # out = color.label2rgb(segments_np, img, kind='avg', bg_label=0)
        # out = segmentation.mark_boundaries(out, segments_np, (0, 0, 0))
        # plt.imshow(out)
        # plt.show()

        with torch.no_grad():

            
            seg_ = segments.cuda().reshape(-1).long()
            mask = torch.tensor(msk, device='cuda').reshape(-1)

            
            # print(mask.size(), seg_.size(), seg_.max())
            seq_mask = scatter(mask, seg_, reduce='mean')
        
        
        
        
        seq_mask = seq_mask.reshape(-1)
        

    
        plt_image = seq_mask.detach().cpu().numpy().reshape(-1)[segments.reshape(-1)].reshape([msk.shape[2], msk.shape[3]])
        plt_image = np.ravel(plt_image)
            

        msk = np.ravel(msk)
        y_temp = (plt_image >= 0.5).astype(np.float)
        tp = np.sum((y_temp * msk))
        # avoid prec becomes 0
        prec, recall = (tp + 1e-10) / (np.sum(y_temp) + 1e-10), (tp + 1e-10) / (np.sum(msk) + 1e-10)
        beta_square = 0.3
        f_score = (1 + beta_square) * prec * recall / (beta_square * prec + recall)
        ious.append(f_score)
    x_label.append(f'SZ {s}, EC {e}')
    all_ious.append(np.mean(ious))

# plt.plot(segment_numbers, all_ious, label=f'SZ {s}, EC {e}')
plt.bar(list(range(len(all_ious))), all_ious, align='center')



fs = 20
plt.title(f'Segmentation boundary intersection accuracy', fontsize=fs)
plt.xlabel('Segmentations', fontsize=fs)
plt.ylabel('Intersection Accuracy', fontsize=fs)
# plt.xscale('log')
plt.xticks(list(range(len(all_ious))), x_label, fontsize=fs, rotation=45)
plt.yticks(fontsize=fs)
plt.ylim((0.95, 1.0))
plt.tight_layout()
plt.show()
# plt.savefig(f'Slic_zero.jpg')
    

