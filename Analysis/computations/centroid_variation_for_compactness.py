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
segment_number = 3136
img_size = 448
compactness = [0.1, 10]

num_images = 3000

H = W = int(segment_number**0.5)
centroids_H = np.linspace((img_size/H)/2, img_size-(img_size/H)/2, H)
centroids_W = np.linspace((img_size/W)/2, img_size-(img_size/W)/2, W)
xv, yv = np.meshgrid(centroids_W, centroids_H)
centroids = np.stack((xv, yv), axis=2).reshape(-1, 2)


for compact in tqdm(compactness):
    all_distances = []

    for file in tqdm(os.listdir(dataset_images)[:num_images]):
        name = file.split('.jpg')[0]
        image = os.path.join(dataset_images, name+'.jpg')
        mask = os.path.join(masks, name+'.png')

        img = Image.open(image)
        msk = Image.open(mask)
        img = img.convert('RGB').resize((img_size, img_size))
        msk = msk.convert('L').resize((img_size, img_size))
        img = np.array(img)
        msk = np.array(msk)
 
      
        
        
        
        segments = slic(img, n_segments=segment_number,
        compactness=compact,
        max_num_iter=10,
        convert2lab=True,
        enforce_connectivity=False,
        slic_zero=True)
        
        regions = regionprops_table(segments, img, properties=('label', 'centroid'))
        
        # plt.scatter(regions['centroid-1'], regions['centroid-0'])
        # plt.show()

        for label, y, x in zip(regions['label'], regions['centroid-0'], regions['centroid-1']):
            all_distances.append(np.linalg.norm(centroids[label-1] - [x, y]))
    plt.hist(all_distances, label=f'Compactness {compact}')
    print(f'Compactness {compact}, Avg Distance {np.mean(all_distances)}')
plt.legend()
plt.show()