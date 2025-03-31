import os
import matplotlib.pyplot as plt
import cv2
from skimage import io
from skimage.segmentation import mark_boundaries, slic, quickshift
from skimage.measure import regionprops_table
import numpy as np
from PIL import Image
from tqdm import tqdm
import pickle
import time
from fast_slic.avx2 import SlicAvx2
import torch
from torch_geometric.utils import scatter
import torch.nn.functional as F
from torch_scatter import scatter_std


dataset_images = '/home/eddie/Datasets/DUTS/DUTS-TR/Image'
masks = '/home/eddie/Datasets/DUTS/DUTS-TR/Mask'
# segment_numbers = [224, 448]# , 300
# segment_numbers = [100, 200, 300, 400, 500, 600, 800, 1000, 1500, 3000, 10000, 45000, 90000]
segment_numbers = [3136]
compactness = [0.1, 1, 10, 50]
d= {}
d['segment_numbers'] = segment_numbers
num_images = 3000
use_pickle = False


precs = np.zeros([4, 256])
recalls = np.zeros([4, 256])

file_count = 0 
for file in tqdm(os.listdir(dataset_images)[:num_images]):
    file_count += 1
    name = file.split('.jpg')[0]
    image = os.path.join(dataset_images, name+'.jpg')
    mask = os.path.join(masks, name+'.png')

    img = Image.open(image)
    msk = Image.open(mask)
    img = img.convert('RGB').resize((320, 320))
    msk = msk.convert('L').resize((320, 320))
    img = np.array(img)
    msk = np.array(msk)
    
    msk[msk>125] = 255
    msk[msk<=125] = 0

    empty_background = np.zeros_like(msk)

    msk_boundaries = np.sum(mark_boundaries(empty_background, msk), axis=2)

    msk[msk<=125] = 0
    msk[msk>125] = 1
    
    
    all_fscores = []
    all_imgs = []
    all_masks = []
    for index, compact in enumerate(compactness):
        segments = slic(img, n_segments=1024,
        compactness=compact,
        max_num_iter=10,
        convert2lab=True,
        enforce_connectivity=False,
        slic_zero=False)
        
        
        regions = regionprops_table(segments, img, properties=('label', 'centroid', 'area', 'intensity_mean',
                                                                                'coords',))
        
        seq_mask = np.zeros([max(regions['label'])])
        # assert len(regions['label']) == max(regions['label']), 'Wrong number of labels'

        for ind, coord in zip(regions['label'], regions['coords']):
            seq_mask[ind-1] = np.sum(msk[coord[:, 0], coord[:, 1]])/len(coord[:, 0])

        plt_image = seq_mask[segments-1].reshape([img.shape[0], img.shape[1]])


        # avoid prec becomes 0
        prec, recall = np.zeros(256), np.zeros( 256)
        pred = np.ravel(plt_image)
        mask = np.ravel(msk)
        thlist = np.linspace(0, 1 - 1e-10, 256)
        for j in range(256):
            y_temp = (pred >= thlist[j])
            tp = (y_temp * mask).sum(-1)
            # avoid prec becomes 0
            prec[j], recall[j] = (tp + 1e-10) / (y_temp.sum(-1) + 1e-10), (tp + 1e-10) / (mask.sum(-1) + 1e-10)
        # (batch, threshold)
        precs[index] += prec
        recalls[index] += recall
        beta_square = 0.3
        f_score = (1 + beta_square) * prec * recall / (beta_square * prec + recall)
        all_fscores.append(np.max(f_score))
        all_imgs.append(mark_boundaries(img, segments-1))
        all_masks.append(plt_image)

    avg_prec = precs/file_count
    avg_recall = recalls/file_count
    beta_square = 0.3
    avg_f_score = (1 + beta_square) * avg_prec * avg_recall / (beta_square * avg_prec + avg_recall)
    print('Running fscores', avg_f_score.max(1))
        
    if np.max(all_fscores) == all_fscores[2]:
        fig, ax = plt.subplots(2, 4, figsize=(16, 4), clear=True)
        for index, (img, mask, fscore) in enumerate(zip(all_imgs, all_masks, all_fscores)):
            ax[0,index].imshow(img)
            ax[1,index].imshow(mask, cmap='gray')
            ax[0,index].set_title(f'Compactness {compactness[index]} F-score {"{:.3f}".format(fscore)}')
        plt.show()

    
    
        

