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
from fast_slic.avx2 import SlicAvx2


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

data_dir = '/mnt/hdd/Datasets/DUTS/DUTS-TR/Image/ILSVRC2012_test_00000004.jpg'
all_distances = []
heights = []
widths = []

img = Image.open(data_dir)
img = img.convert('RGB')
img = img.resize((256,256), resample=Image.BILINEAR)


policy = transforms.AutoAugmentPolicy.IMAGENET
augmenter = transforms.AutoAugment(policy)
imgs = [augmenter(img) for _ in range(4)]


fig, ax = plt.subplots(2, 2)
for ind, img in enumerate(imgs):

    img_np = np.array(img)

    num_seg = 400


    slic = SlicAvx2(num_components=num_seg, compactness=10)
    segments = slic.iterate(img_np)+1
    row = ind//2
    col = ind%2
    ax[row, col].imshow(mark_boundaries(img_np, segments))

plt.show()


