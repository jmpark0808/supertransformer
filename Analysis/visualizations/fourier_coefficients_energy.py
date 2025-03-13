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
from math import pi, atan2

dataset_images = '/home/eddie/Datasets/DUTS/DUTS-TR/Image'
masks = '/home/eddie/Datasets/DUTS/DUTS-TR/Mask'



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

class clockwise_angle_and_distance():
    '''
    A class to tell if point is clockwise from origin or not.
    This helps if one wants to use sorted() on a list of points.

    Parameters
    ----------
    point : ndarray or list, like [x, y]. The point "to where" we g0
    self.origin : ndarray or list, like [x, y]. The center around which we go
    refvec : ndarray or list, like [x, y]. The direction of reference

    use: 
        instantiate with an origin, then call the instance during sort
    reference: 
    https://stackoverflow.com/questions/41855695/sorting-list-of-two-dimensional-coordinates-by-clockwise-angle-using-python

    Returns
    -------
    angle
    
    distance
    

    '''
    def __init__(self, origin):
        self.origin = origin

    def __call__(self, point, refvec = [0, 1]):
        if self.origin is None:
            raise NameError("clockwise sorting needs an origin. Please set origin.")
        # Vector between point and the origin: v = p - o
        vector = [point[0]-self.origin[0], point[1]-self.origin[1]]
        # Length of vector: ||v||
        lenvector = np.linalg.norm(vector[0] - vector[1])
        # If length is zero there is no angle
        if lenvector == 0:
            return -pi, 0
        # Normalize vector: v/||v||
        normalized = [vector[0]/lenvector, vector[1]/lenvector]
        dotprod  = normalized[0]*refvec[0] + normalized[1]*refvec[1] # x1*x2 + y1*y2
        diffprod = refvec[1]*normalized[0] - refvec[0]*normalized[1] # x1*y2 - y1*x2
        angle = atan2(diffprod, dotprod)
        # Negative angles represent counter-clockwise angles so we need to 
        # subtract them from 2*pi (360 degrees)
        if angle < 0:
            return 2*pi+angle, lenvector
        # I return first the angle because that's the primary sorting criterium
        # but if two vectors have the same angle then the shorter distance 
        # should come first.
        return angle, lenvector


def fourier_descriptors(region, N, label):
    # fig, ax = plt.subplots(1, 2)
    region = (region*255).astype(np.uint8)
    
    # ax[0].imshow(region, cmap='gray')
    contour, hierchy = cv2.findContours(region, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    if len(contour) > 1:
        # if label == 177:
        #     plt.imshow(region)
        #     plt.show()
        
    
        list_of_pts = []
        for ctr in contour:
            list_of_pts += [pt[0] for pt in ctr]

        center_pt = np.array(list_of_pts).mean(axis = 0) # get origin
        clock_ang_dist = clockwise_angle_and_distance(center_pt) # set origin
        
        list_of_pts = sorted(list_of_pts, key=clock_ang_dist) # use to sort

        points = np.array(list_of_pts).reshape((-1, 2)).astype(np.int32)
        # points = cv2.convexHull(points)[:, 0, :]
        # perimeter = cv2.arcLength(points, True)
        # points = cv2.approxPolyDP(points, 0.02 * perimeter, True)[:, 0, :]
        indices_y = np.argwhere(points[:, 1]==np.min(points[:, 1])) # smallest y
        indices_x = np.argmin(points[indices_y, 0])
        points = np.roll(points, -indices_y[indices_x], axis=0)
        # plt.plot(points[:, 0], points[:, 1])
        # plt.scatter(points[0, 0], points[0, 1], c='red')
        # plt.scatter(points[1, 0], points[1, 1], c='green')
        # plt.scatter(points[2:, 0], points[2:, 1], c='blue')
        # plt.title('merged')
        # plt.show()
    else:
    # assert(len(contour)==1)
        points = contour[0][:, 0, :]
    # if len(contour) > 1:
    #     plt.plot(ctr[:, 0], ctr[:, 1])
    #     plt.scatter(ctr[0, 0], ctr[0, 1], c='red')
    #     plt.scatter(ctr[1, 0], ctr[1, 1], c='green')
    #     plt.scatter(ctr[2:, 0], ctr[2:, 1], c='blue')
    #     plt.title('merged')
    #     plt.show()
    # else:
    #     plt.plot(ctr[:, 0], ctr[:, 1])
    #     plt.scatter(ctr[0, 0], ctr[0, 1], c='red')
    #     plt.scatter(ctr[1, 0], ctr[1, 1], c='green')
    #     plt.scatter(ctr[2:, 0], ctr[2:, 1], c='blue')
    #     plt.show()
    
    
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
    return xi, yi, orig_xi, orig_yi, fourier_result

# fig, ax = plt.subplots(1, 2, figsize=(10, 10))
resample_points = int(((448**2)//784)**0.5)*4
'''
coeff = 10
all_fourier_results = []
for file in tqdm(os.listdir(dataset_images)[:100]):
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
    enforce_connectivity=True,
    slic_zero=False)-1

    # np.save('sample_segment.npy', segments)

    regions = regionprops_table(segments, img, properties=('label', 'centroid'))
    centroids_x = regions['centroid-1']
    centroids_y = regions['centroid-0']
    labels = regions['label']
    fig, ax = plt.subplots(1, 4, figsize=(20, 5))
    im = np.zeros((448, 448, 3))
    grid = np.arange(0, 784).reshape(28, 28).reshape(4, 7, 4, 7).transpose((0, 2, 1, 3)).reshape(16, 49)
    # assert(len(labels) == np.max(labels))
    # for indices in grid:
    #     plt.scatter(centroids_x[indices], -centroids_y[indices])
    #     plt.ylim(-448, 0)
    #     plt.xlim(0, 448)
    #     plt.show()


    # plt.scatter(centroids_x, -centroids_y)
    
    # for x, y, l in zip(centroids_x, -centroids_y, labels):
    #     plt.text(x+1, y+1, s=l-1 )
    # plt.show()  

                    
    
   
    for label in range(784):
        if np.sum(segments == label) > 0:
            regions = segments == label
           
            xi, yi, og_xi, og_yi, fourier_result= fourier_descriptors(regions, 5, label)
            all_fourier_results.append(fourier_result)


all_fourier_results = np.array(all_fourier_results)
np.save('all_fourier_results.npy', all_fourier_results)

''' 

all_fourier_results = np.load('/home/eddie/waterloo/supertransformer/Analysis/all_fourier_results.npy')

all_fourier_results = all_fourier_results[:, 1:]

new_array = []
for i in range(all_fourier_results.shape[1]):
    if i % 2 == 0: # even
        new_array.append(all_fourier_results[:, 0])
        all_fourier_results = all_fourier_results[:, 1:]
    else: # odd
        new_array.append(all_fourier_results[:, -1])
        all_fourier_results = all_fourier_results[:, :-1]
new_array = np.array(new_array)
print(new_array[0, 0])
new_array = abs(new_array)
print(new_array[0, 0])
new_array = np.cumsum(new_array, axis=0)

new_array = new_array/new_array[-1, :]
new_array_mean = np.mean(new_array, axis=1)
new_array_std = np.std(new_array, axis=1)
print(new_array_std)
plt.plot(new_array_mean)
plt.fill_between(np.arange(0, 63), np.subtract(new_array_mean, new_array_std), np.add(new_array_mean, new_array_std), alpha=0.2)
plt.xlabel('Coefficient Index', fontsize=15)
plt.ylabel('Normalized Cumulative Energy', fontsize=15)
# plt.axvline(10, 0, 1, linestyle='--', c='red', label='Plausible threshold to capture most energy')
plt.legend(loc='lower right')
plt.savefig('/mnt/hdd/Figures/SuperFormer/fourier_coefficients_energy.pdf', format='pdf')
plt.show()
    
