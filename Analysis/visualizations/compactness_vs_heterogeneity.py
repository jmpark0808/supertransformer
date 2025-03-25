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
compactness = [0.1, 1, 10, 100]


y_indices, x_indices = np.meshgrid(np.arange(56), np.arange(56), indexing='ij')
patch_centroids = np.stack([x_indices, y_indices], axis=-1).reshape(3136, 2)+1.5
average_norms = []
for compact in compactness:
    count = 0 
    all_norms = 0 
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

        segments = slic(img, n_segments=3136,
        compactness=compact,
        max_num_iter=10,
        convert2lab=True,
        enforce_connectivity=False,
        slic_zero=False)
        
        regions = regionprops_table(segments, img, properties=('label', 'centroid'))

        for label, y, x in zip(regions['label'], regions['centroid-0'], regions['centroid-1']):
            all_norms += np.linalg.norm(patch_centroids[label-1] - np.array([x, y]))
            count += 1

    average_norms.append(all_norms/count)
   



# Plot heatmap
plt.plot(compactness, average_norms)
plt.title("Centroid Displacement as a Function of Compactness")
plt.ylabel('Average Pixel Displacement (L2 norm)')
plt.xlabel('Compactness')
plt.xticks([0.1, 1, 10, 100])
plt.tight_layout()
plt.savefig('/mnt/hdd/Figures/SuperFormer/centroid_displacement.pdf', format='pdf')
plt.show()