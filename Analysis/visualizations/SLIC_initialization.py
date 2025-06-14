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



fig, ax = plt.subplots(1, 3, figsize=(15, 6.5))



        
for file in tqdm(os.listdir(dataset_images)):
    name = file.split('.jpg')[0]
    image = os.path.join(dataset_images, name+'.jpg')
    mask = os.path.join(masks, name+'.png')

    img = Image.open(image)
    msk = Image.open(mask)
    img = img.convert('RGB').resize((448,448))
    msk = msk.convert('L').resize((448, 448))
    img = np.array(img)
    msk = np.array(msk)
    
    
    msk[msk>125] = 255
    msk[msk<=125] = 0

    empty_background = np.zeros_like(msk)

    msk_boundaries = np.sum(mark_boundaries(empty_background, msk), axis=2)

    msk[msk<=125] = 0
    msk[msk>125] = 1
    
    label = 200
    
    segments1 = slic(img, n_segments=768,
    compactness=10,
    max_num_iter=1,
    convert2lab=True,
    enforce_connectivity=False,
    slic_zero=False)
    seg1_mask = (segments1 == label)[..., None]

    img1 = mark_boundaries(img, segments1)
    red_overlay = np.zeros_like(img1)
    red_overlay[..., 0] = 1.0 
    img1 = np.where(seg1_mask, red_overlay, img1)
    ax[0].imshow(img1, aspect='equal')
    ax[0].axis('off')
    ax[0].set_title('Initial SLIC Segmentation', fontsize=20)
    segments2 = slic(img, n_segments=768,
    compactness=10,
    max_num_iter=10,
    convert2lab=True,
    enforce_connectivity=False,
    slic_zero=False)
    seg2_mask = (segments2 == label)[..., None]

    img2 = mark_boundaries(img, segments2)
    green_overlay = np.zeros_like(img2)
    green_overlay[..., 1] = 1.0 
    img2 = np.where(seg2_mask, green_overlay, img2)

    ax[1].imshow(img2, aspect='equal')
    ax[1].axis('off')
    ax[1].set_title('Final SLIC Segmentation', fontsize=20)


     # Create binary masks

    
    mask1 = (segments1 == label)
    mask2 = (segments2 == label)

    # Compute intersection and union
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    iou = intersection.sum() / union.sum() if union.sum() != 0 else 0.0

    # Get bounding box of the union
    coords = np.argwhere(union)

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1

    # Apply padding and clip
    padding = 2
    y0 = max(0, y0 - padding)
    x0 = max(0, x0 - padding)
    y1 = min(segments1.shape[0], y1 + padding)
    x1 = min(segments1.shape[1], x1 + padding)

    # Create RGB visualization
    vis = np.ones((*segments1.shape, 3), dtype=np.float32)
    vis[mask1] = [1, 0, 0]        # Red for seg1
    vis[mask2] = [0, 1, 0]        # Green for seg2
    vis[intersection] = [1, 1, 0] # Yellow for overlap

    # Crop visualization
    vis_cropped = vis[y0:y1, x0:x1]
    
    vis_cropped = vis_cropped[1:, :, :]
    asp = np.diff(ax[0].get_xlim())[0] / np.diff(ax[0].get_ylim())[0]
    
    ax[2].imshow(vis_cropped)
    ax[2].axis('off')
    ax[2].set_title('IoU', fontsize=20)
    ax[2].set_aspect('equal')

    fig.tight_layout()

    fig.savefig('/mnt/hdd/Figures/SuperFormer/SLIC_init.pdf', format='pdf')
    assert(0)
    
    
      