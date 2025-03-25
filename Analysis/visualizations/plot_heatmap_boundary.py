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


num_images = 3000

segmentation_maps = []

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

segmentation_maps = np.stack(segmentation_maps, axis=0)  

# Compute consistency (fraction of times each pixel stays in the same superpixel)
consistency_map = np.zeros_like(segmentation_maps[0], dtype=float)
num_runs = segmentation_maps.shape[0]

for i in range(num_runs):
    for j in range(i + 1, num_runs):
        consistency_map += (segmentation_maps[i] == segmentation_maps[j])

# Normalize (convert to fraction)
consistency_map /= (num_runs * (num_runs - 1) / 2)

# Plot heatmap
plt.imshow(consistency_map, cmap="coolwarm")
plt.colorbar(label="Superpixel Stability")
plt.title("Superpixel Consistency Heatmap")
plt.axis('off')
plt.tight_layout()
plt.savefig('/mnt/hdd/Figures/SuperFormer/boundary_heatmap.pdf', format='pdf')
plt.show()