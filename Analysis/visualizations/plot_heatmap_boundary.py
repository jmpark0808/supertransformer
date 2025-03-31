import numpy as np
import matplotlib.pyplot as plt

# Assume multiple segmentation maps: (num_runs, H, W)
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
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde as kde
from matplotlib.colors import Normalize
from matplotlib import cm
from matplotlib.patches import Rectangle

dataset_images = '/home/eddie/Datasets/DUTS/DUTS-TR/Image'
masks = '/home/eddie/Datasets/DUTS/DUTS-TR/Mask'

def compute_superpixel_iou(seg1, seg2, num_seg):
    """
    Computes the Intersection over Union (IoU) for each superpixel label between two segmentation maps.
    
    Args:
        seg1 (torch.Tensor): First segmentation map of shape (H, W)
        seg2 (torch.Tensor): Second segmentation map of shape (H, W)
    
    Returns:
        torch.Tensor: IoU scores of shape (N,), where N is the number of unique superpixels in seg1.
    """

    iou_scores = np.zeros([num_seg])
    
    for i, label in enumerate(range(num_seg)):
        mask1 = seg1 == label  # Binary mask for the current superpixel in seg1
        mask2 = seg2 == label  # Binary mask for the corresponding superpixel in seg2
        
        intersection = np.sum(mask1 & mask2).astype(np.float32)
        union = np.sum(mask1 | mask2).astype(np.float32)
        
        iou_scores[i] = intersection / union if union > 0 else 0.0
    
    return iou_scores


num_images = 3000
segmentation_maps = []
mask_maps = []
all_ious = []

grid_segments = slic(np.zeros([448, 448, 3]), n_segments=784,
    compactness=10,
    max_num_iter=1,
    convert2lab=True,
    enforce_connectivity=False,
    slic_zero=False)



for file in tqdm(os.listdir(dataset_images)[:num_images]):
    name = file.split('.jpg')[0]
    image = os.path.join(dataset_images, name+'.jpg')
    mask = os.path.join(masks, name+'.png')

    img = Image.open(image)
    msk = Image.open(mask)
    img = img.convert('RGB').resize((448, 448))
    msk = msk.convert('L').resize((448, 448))
    img = np.array(img)
    msk = np.array(msk)
    
    
    msk[msk>125] = 255
    msk[msk<=125] = 0

    segments = slic(img, n_segments=784,
    compactness=10,
    max_num_iter=10,
    convert2lab=True,
    enforce_connectivity=False,
    slic_zero=False)
    segmentation_maps.append(segments)
    msk = msk/255.
    mask_maps.append(msk)

    iou = compute_superpixel_iou(segments-1, grid_segments-1, 784)
    all_ious.append(iou)

iou_means = 1-np.mean(np.array(all_ious), axis=0)



segmentation_maps = np.stack(segmentation_maps, axis=0)  
mask_maps = np.stack(mask_maps, axis=0)  
# Compute consistency (fraction of times each pixel stays in the same superpixel)
consistency_map = np.zeros_like(segmentation_maps[0], dtype=float)
num_runs = segmentation_maps.shape[0]

mask_map = mask_maps.sum(0)/mask_maps.shape[0]

for i in tqdm(range(num_runs)):
    for j in range(i + 1, num_runs):
        consistency_map += (segmentation_maps[i] == segmentation_maps[j])

# Normalize (convert to fraction)
consistency_map /= (num_runs * (num_runs - 1) / 2)

# Plot heatmap
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

im0 = ax[0].imshow(iou_means.reshape(28, 28), cmap='coolwarm')

# im0 = ax[0].imshow(consistency_map, cmap="jet")
cbar0 = plt.colorbar(im0,fraction=0.046, pad=0.04)
cbar0.set_label('IoU Complement', fontsize=15)
# ax[0].set_title("Superpixel Segmentation Inconsistency", fontsize=15, pad=20)
ax[0].axis('off')

im1 = ax[1].imshow(mask_map, cmap="coolwarm")
ax[1].axis('off')
cbar1 = plt.colorbar(im1, fraction=0.046, pad=0.04)
cbar1.set_label("Average Saliency", fontsize=15)
# ax[1].set_title('Average Saliency', fontsize=15, pad=20)
fig.tight_layout()
fig.savefig('/mnt/hdd/Figures/SuperFormer/boundary_heatmap.pdf', format='pdf')
plt.show()