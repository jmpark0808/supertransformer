# load image 
import sys
from threading import local
sys.path.insert(0, '/home/eddie/waterloo/supertransformer')
from torchvision import transforms, datasets
import torch
from PIL import Image
from Blocks import blocks
import numpy as np
from skimage.segmentation import slic
from skimage.measure import regionprops_table
from skimage.segmentation import mark_boundaries
from skimage.feature import local_binary_pattern
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os
from numpy_superpixel import SLICProcessor
import time
from sklearn.metrics.pairwise import euclidean_distances
from tqdm import tqdm
from scipy import sparse as sp
from dataset.constants import *
from scipy.spatial.distance import pdist, squareform


def shape(region):
    # note the ddof arg to get the sample var if you so desire!
    centroid = np.mean(np.nonzero(region),axis=1)
    print(centroid)
    coords = np.nonzero(region)
    normalized_coords = np.stack([coords[0]-centroid[0], coords[1]-centroid[1]], axis=1)
    rho = np.linalg.norm(normalized_coords, axis=1)
    phi = np.arctan2(normalized_coords[:, 0], normalized_coords[:, 1])*180/np.pi+180
    radii = []
    degrees = []

    chunk = CHUNK
    
    for ind, degree in enumerate(range(0, 360, chunk)):
        try:
            radii.append(np.max(rho[(degree<=phi) & (phi<degree+chunk)]))
            degrees.append(phi[(degree<=phi) & (phi<degree+chunk)][np.argmax(rho[(degree<=phi) & (phi<degree+chunk)])])
        except: 
            pass

    if len(radii) != NUM_CHUNK and np.sum(region) > 10:
        print(len(radii))
        print(region)
        print(normalized_coords)
        plt.imshow(region.astype(np.int16), cmap='gray', vmin=0, vmax=1)
        for ind, radius in enumerate(radii):
            degree = (degrees[ind]-180)*np.pi/180.
            x = radius * np.cos(degree)
            y = radius * np.sin(degree)
            plt.plot([centroid[0], centroid[0]+x], [centroid[1], centroid[1]+y])
        plt.show()
        assert(0)
        

    # if 0.6 < np.sum(region)/region.size < 0.8:
    #     print(len(radii))
    #     print(region)
    #     print(normalized_coords)
    #     plt.imshow(region.astype(np.int16), cmap='gray', vmin=0, vmax=1)
    #     for ind, radius in enumerate(radii):
    #         degree = (degrees[ind]-180)*np.pi/180.
    #         x = radius * np.cos(degree)
    #         y = radius * np.sin(degree)
    #         plt.plot([centroid[0], centroid[0]+x], [centroid[1], centroid[1]+y])
    #     plt.show()
    #     assert(0)

    return np.array(radii)


def polarize(region):
    # note the ddof arg to get the sample var if you so desire!
    centroid = np.mean(np.nonzero(region),axis=1)
    coords = np.nonzero(region)
    normalized_coords = np.stack([coords[0]-centroid[0], coords[1]-centroid[1]], axis=1)
    rho = np.linalg.norm(normalized_coords, axis=1)
    phi = np.arctan2(normalized_coords[:, 0], normalized_coords[:, 1])*180/np.pi+180
    radii_max = np.zeros([NUM_CHUNK, 2])
    radii_min = np.zeros([NUM_CHUNK, 2])

    chunk = CHUNK
    
    for ind, degree in enumerate(range(0, 360, chunk)):
        try:
            radii_max[ind] = normalized_coords[np.argmax(np.where((degree<=phi) & (phi<degree+chunk), rho, np.zeros_like(rho)))]
        except: 
            pass
        
        try:
            radii_min[ind] = normalized_coords[np.argmin(np.where((degree<=phi) & (phi<degree+chunk), rho, np.inf*np.ones_like(rho)))]
        except: 
            pass
        
        
    return np.concatenate((radii_max, radii_min), axis=0)


def hist(region, intensities):
    # note the ddof arg to get the sample var if you so desire!
    (hist, _) = np.histogram(intensities[region], bins=BINS, range=(0, 255), density=False)
    return hist



def embed(region, intensities):
    # note the ddof arg to get the sample var if you so desire!
    cut_out = np.zeros([24, 24])
    cut_out[np.nonzero(region)] = intensities[np.nonzero(region)]
    return (cut_out.reshape(-1))
    
def image_stdev(region, intensities):
    # note the ddof arg to get the sample var if you so desire!
    return np.std(intensities[region])

data_dir = '/mnt/hdd/Datasets/SegTrackv2/JPEGImages/bird_of_paradise'

for file in tqdm(os.listdir(data_dir)):
    img = Image.open(os.path.join(data_dir, file))
    img = img.convert('RGB')
    img = img.resize((224, 224), resample=Image.BILINEAR)

    img_np = np.array(img).astype(np.float32)/255.

    num_seg = 625


    img_size = img_np.shape[1]

    start = time.time()
    segments = slic(img, n_segments=num_seg,
        compactness=10.0,
        max_num_iter=3,
        convert2lab=True,
        enforce_connectivity=False,
        slic_zero=False)
    vs_right = np.vstack([segments[:,:-1].ravel(), segments[:,1:].ravel()])
    vs_below = np.vstack([segments[:-1,:].ravel(), segments[1:,:].ravel()])
    vs_diagonal_r = np.vstack([segments[:-1,:-1].ravel(), segments[1:,1:].ravel()])
    vs_diagonal_l = np.vstack([segments[1:,:-1].ravel(), segments[:-1,1:].ravel()])
    bneighbors = np.unique(np.hstack([vs_right, vs_below, vs_diagonal_r, vs_diagonal_l]), axis=1)
    end = time.time()

    regions = regionprops_table(segments, intensity_image=img_np, properties=('label', 'centroid', 'area', 'intensity_mean',
                                                                                'extent', 'coords', 'eccentricity'))#, polarize])
                
    seq_len = len(regions['label'])
    features = np.zeros([num_seg, 11])
    seq_mask = np.zeros([num_seg])
    label = regions['label']
    features[label-1, 0] = regions['centroid-0']
    features[label-1, 1] = regions['centroid-1']
    features[label-1, 2] = regions['area'] / (img_size**2)
    features[label-1, 3] = regions['intensity_mean-0']/255.
    features[label-1, 4] = regions['intensity_mean-1']/255.
    features[label-1, 5] = regions['intensity_mean-2']/255.
    features[label-1, 6] = regions['extent']
    features[label-1, 7] = regions['eccentricity']



    neighbor_array = np.zeros([num_seg, num_seg])
    # eye = np.eye(self.num_seg)
    neighbor_array[bneighbors[0]-1, bneighbors[1]-1] = 1
    neighbor_array[bneighbors[1]-1, bneighbors[0]-1] = 1
    # neighbor_array -= eye
    smoothed_images = np.zeros_like(img_np)
    for coord, r, g, b in zip(regions['coords'], regions['intensity_mean-0'], regions['intensity_mean-1'], regions['intensity_mean-2']):
        for c in coord:
            smoothed_images[c[0], c[1], 0] = r
            smoothed_images[c[0], c[1], 1] = g
            smoothed_images[c[0], c[1], 2] = b

    # plt.imshow(mark_boundaries(smoothed_images, segments))
    # plt.axis('off')
    # plt.savefig(os.path.join(data_dir, file+'_smoothed.jpg'))
    plt.imsave(os.path.join(data_dir, file+'_smoothed.jpg'), smoothed_images)


    # assert(0)

    
# print(np.min(heights), np.max(heights))
# print(np.min(widths), np.max(widths))