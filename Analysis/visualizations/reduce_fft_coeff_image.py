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

dataset_images = '/mnt/hdd/Datasets/DUTS/DUTS-TR/Image'
masks = '/mnt/hdd/Datasets/DUTS/DUTS-TR/Mask'
# segment_numbers = [224, 448]# , 300
# segment_numbers = [100, 200, 300, 400, 500, 600, 800, 1000, 1500, 3000, 10000, 45000, 90000]
segment_numbers = [3136]
compactness = [10]
d= {}
d['segment_numbers'] = segment_numbers
num_images = 3000
use_pickle = False


def resample_2d(points, N):

    xc = points[:, 0].tolist() + [points[0, 0]]
    yc = points[:, 1].tolist() + [points[0, 1]]

    dx = np.diff(xc)
    dy = np.diff(yc)

    dS = np.sqrt(dx**2+dy**2)
    dS = np.array([0]+dS.tolist())

    d = np.cumsum(dS)

    perim = d[-1]

    ds = perim/N
    dSi = ds*np.arange(0, N)
    dSi[-1] = dSi[-1] - 0.005

    xi = np.interp(dSi, d, xc)
    yi = np.interp(dSi, d, yc)
    return xi, yi


def fourier_descriptors(region, N):
    # fig, ax = plt.subplots(1, 2)
    region = (region*255).astype(np.uint8)
    # ax[0].imshow(region, cmap='gray')
    contour, hierarchy = cv2.findContours(region, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    points = cnt[:, 0, :], for cnt in contour
    # ax[1].scatter(points[:, 0], points[:, 1])
    # plt.show()
    xi, yi = resample_2d(points, resample_points)
    
    orig_xi, orig_yi = np.copy(xi), np.copy(yi)
    contour_array = np.stack((xi, yi), axis=1)


    contour_complex = np.empty(contour_array.shape[:-1], dtype=complex)
    contour_complex.real = contour_array[:, 0]
    contour_complex.imag = contour_array[:, 1]
    fourier_result = np.fft.fft(contour_complex)

    truncated_fourier_result = np.copy(fourier_result)
    truncated_fourier_result[1+N:-N] = 0
    inverse_fourier_result = np.fft.ifft(truncated_fourier_result)
    contour_reconstruct = np.array(
            [inverse_fourier_result.real, inverse_fourier_result.imag])
    contour_reconstruct = np.transpose(contour_reconstruct)

    xi = contour_reconstruct[:, 0]
    yi = contour_reconstruct[:, 1]
    # return np.array(amp)
    return xi, yi, orig_xi, orig_yi

# fig, ax = plt.subplots(1, 2, figsize=(10, 10))
resample_points = int(((448**2)//784)**0.5)*4

coeff = 10

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

    empty_background = np.zeros_like(msk)

    msk_boundaries = np.sum(mark_boundaries(empty_background, msk), axis=2)

    msk[msk<=125] = 0
    msk[msk>125] = 1
    
    

    segments = slic(img, n_segments=784,
    compactness=10,
    max_num_iter=10,
    convert2lab=True,
    enforce_connectivity=False,
    slic_zero=False)-1


    for label in range(784):
        regions = segments == label
        xi, yi, og_xi, og_yi= fourier_descriptors(regions, 30)
        xi = np.append(xi, xi[0])
        yi = np.append(yi, yi[0])
        og_xi = np.append(og_xi, og_xi[0])
        og_yi = np.append(og_yi, og_yi[0])
        plt.plot(og_xi, og_yi)

    plt.show()

  
    

    
    
    

            
  