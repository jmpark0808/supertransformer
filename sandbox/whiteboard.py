import os
import torch.utils.data as data
import torchvision.transforms as transforms
from collections import defaultdict
import numpy as np
import torch
from PIL import Image, ImageCms
from skimage.segmentation import slic, mark_boundaries
from skimage.measure import regionprops_table
from skimage.feature import local_binary_pattern
from sklearn.metrics.pairwise import euclidean_distances
from skimage import color
import pytorch_lightning as pl
from fast_slic.avx2 import SlicAvx2
from dataset.constants import *
import matplotlib.pyplot as plt
from scipy import sparse as sp
from scipy.spatial.distance import pdist, squareform
import scipy.sparse.linalg as spla
from dataset.attributes import *
from torch.utils.data import DataLoader
from dataset.randaugment import RandAugment
from pathlib import Path
from tqdm import tqdm
from dataset.fft_transform import *
from dataset.moments_transform import *
import torch.nn.functional as F
from util.util import merge_contours, compute_central_moments
from torchvision.transforms import ToTensor
from dataset.imagenet import ImageNetDataset


size = 224
num_seg = 3136
compactness = 1
coeff = 10

resample_points = int(((size**2)//num_seg)**0.5)*4
        

        
def fourier_descriptors(region):
    moments = compute_central_moments(region)
    region = (region*255).astype(np.uint8)
    contour, hierarchy = cv2.findContours(region, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(contour)>1:
        merged_contour = merge_contours(contour)

        points = np.array(merged_contour).reshape((-1, 2)).astype(np.int32)

        indices_y = np.argwhere(points[:, 1]==np.min(points[:, 1])) # smallest y
        indices_x = np.argmin(points[indices_y, 0])
        points = np.roll(points, -indices_y[indices_x], axis=0)
    else:
        points = contour[0][:, 0, :]
    xi, yi = resample_2d(points, resample_points)
    contour_array = np.stack((xi, yi), axis=1)


    contour_complex = np.empty(contour_array.shape[:-1], dtype=complex)
    contour_complex.real = contour_array[:, 0]
    contour_complex.imag = contour_array[:, 1]
    fourier_result = np.fft.fft(contour_complex)[1:]


    amp = abs(fourier_result)
    phase = np.arctan2(fourier_result.imag, fourier_result.real)

    # return np.array(amp)
    return np.concatenate((amp, phase, moments))
    


def lbp(region, intensities):
    (hist, _) = np.histogram(intensities[region].ravel(),
            bins=np.arange(0, 8+3),
            range=(0, 8+2))
    hist = hist.astype("float")
    # hist /= (hist.sum() + 1e-7)
    return hist
        

img_path = '/home/eddie/Datasets/DUTS/DUTS-TR/Image'
mask_path = '/home/eddie/Datasets/DUTS/DUTS-TR/Mask'

for img, mask in zip(sorted(os.listdir(img_path)), sorted(os.listdir(mask_path))):
    img = Image.open(os.path.join(img_path, img)).convert('RGB').resize((size, size))
    mask =  Image.open(os.path.join(mask_path, mask)).convert('RGB').resize((size, size))

    img_gray = np.array(img.convert('L'))
    img_np = np.array(img)
    mask_np = np.array(mask)/255.
    img_size = img_np.shape

    # img_np = np.ascontiguousarray(np.transpose(img.cpu().numpy()*255, (1, 2, 0))).astype(np.uint8)
        
    # slic = SlicAvx2(num_components=self.num_seg, compactness=self.compactness)
    # segments = slic.iterate(img_np)+1
    segments = slic(img_np, n_segments=num_seg,
        compactness=compactness,
        max_num_iter=10,
        convert2lab=True,
        enforce_connectivity=False,
        slic_zero=False)

    plt.imshow(mark_boundaries(img_np, segments))
    plt.show()

    vs_right = np.vstack([segments[:,:-1].ravel(), segments[:,1:].ravel()])
    vs_below = np.vstack([segments[:-1,:].ravel(), segments[1:,:].ravel()])
    bneighbors, counts = np.unique(np.hstack([vs_right, vs_below]), axis=1, return_counts=True)
    
    

    # for i in range(bneighbors.shape[1]):
    #     if bneighbors[0,i] != bneighbors[1,i]:
    #         edge_attr[bneighbors[0,i]-1, bneighbors[1,i]-1] = counts[i]
            
    lbp_np = local_binary_pattern(img_gray, 8, 1, method='uniform')
    regions_lbp = regionprops_table(segments, intensity_image=lbp_np, extra_properties=[lbp])

    regions = regionprops_table(segments, intensity_image=img_np, properties=('label', 'centroid', 'intensity_mean',
                                                                                'coords', 'area'), extra_properties=[image_stdev, fourier_descriptors])#, polarize])
    
    seq_len = len(regions['label'])
    seq_mask = np.zeros([num_seg])
    label = regions['label']
    # features = np.zeros([self.num_seg, 8+(self.resample_points-1)*2+10])
    

    features = np.zeros([num_seg, 8+((resample_points-1)*2)+8+10])
    
    # for i in range((self.resample_points-1)*2):
    for i in range((resample_points-1)*2+8):
        features[label-1, 8+i] = regions[f'fourier_descriptors-{i}']

    




    features[label-1, 0] = regions['centroid-0']
    features[label-1, 1] = regions['centroid-1']
    
    features[label-1, 2] = regions['intensity_mean-0']/255.
    features[label-1, 3] = regions['intensity_mean-1']/255.
    features[label-1, 4] = regions['intensity_mean-2']/255.
    features[label-1, 5] = regions['image_stdev-0']/255.
    features[label-1, 6] = regions['image_stdev-1']/255.
    features[label-1, 7] = regions['image_stdev-2']/255.

    for ind in range(8+2):
        # features[label-1, ind+8+(self.resample_points-1)*2] = regions_lbp[f'lbp-{ind}']
        features[label-1, ind+8+(resample_points-1)*2+8] = regions_lbp[f'lbp-{ind}']
    
    
    for ind, coord in zip(regions['label'], regions['coords']):
        seq_mask[ind-1] = np.sum(mask_np[coord[:, 0], coord[:, 1]])/len(coord[:, 0])
