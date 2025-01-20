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

def merge_contours(contours):
    """
    Merges multiple contours by connecting their closest points and updating the order of points.

    Parameters:
    - contours: List of contours from OpenCV's findContours, where each contour is a numpy array of shape (n, 1, 2).

    Returns:
    - merged_contour: A single contour combining all input contours.
    """
    # Flatten the contours to remove hierarchy dimension
    contours = [c.reshape(-1, 2) for c in contours]

    while len(contours) > 1:
        # Find the two closest contours
        min_distance = float('inf')
        closest_pair = None

        for i in range(len(contours)):
            for j in range(i + 1, len(contours)):
                contour_a = contours[i]
                contour_b = contours[j]

                # Compute pairwise distances
                distances = np.linalg.norm(
                    contour_a[:, None, :] - contour_b[None, :, :], axis=2)
                min_idx = np.unravel_index(np.argmin(distances), distances.shape)

                distance = distances[min_idx]
                if distance < min_distance:
                    min_distance = distance
                    closest_pair = (i, j, min_idx)

        # Retrieve the indices of the two closest contours and points
        i, j, (idx_a, idx_b) = closest_pair
        contour_a, contour_b = contours[i], contours[j]

        # Reorder contour_a and contour_b to maintain continuity
        contour_a = np.roll(contour_a, -idx_a, axis=0)
        contour_b = np.roll(contour_b, -idx_b, axis=0)

        # Create a connecting line between the two contours
        connecting_line = np.array([contour_a[-1], contour_b[0]])

        # Merge the contours and connecting line
        merged_contour = np.vstack([contour_a, connecting_line, contour_b])

        # Update the contours list
        contours.pop(j)  # Remove second contour (higher index first to avoid indexing issues)
        contours.pop(i)  # Remove first contour
        contours.append(merged_contour)  # Add merged contour back to the list

    # Return the final merged contour
    return contours[0].reshape(-1, 1, 2)


def fourier_descriptors(region, N, label):
    # fig, ax = plt.subplots(1, 2)
    region = (region*255).astype(np.uint8)
    
    # ax[0].imshow(region, cmap='gray')
    contour, hierchy = cv2.findContours(region, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    if len(contour) > 1:
        # if label == 177:
        #     plt.imshow(region)
        #     plt.show()
        
        merged_contour = merge_contours(contour)

        # list_of_pts = []
        # for ctr in contour:
        #     list_of_pts += [pt[0] for pt in ctr]

        # center_pt = np.array(list_of_pts).mean(axis = 0) # get origin
        # clock_ang_dist = clockwise_angle_and_distance(center_pt) # set origin
        
        # list_of_pts = sorted(list_of_pts, key=clock_ang_dist) # use to sort

        points = np.array(merged_contour).reshape((-1, 2)).astype(np.int32)
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
    enforce_connectivity=True,
    slic_zero=False)

    # np.save('sample_segment.npy', segments)

    regions = regionprops_table(segments, img, properties=('label', 'centroid', 'intensity_mean'))
    centroids_x = regions['centroid-1']
    centroids_y = regions['centroid-0']
    labels = regions['label']
    fig, ax = plt.subplots(3, 3, figsize=(10, 10))
    im_plot = np.zeros((448, 448, 3))
    
                    
    im = np.zeros((3, 784))
   
    for label, r, g, b in zip(regions['label'], regions['intensity_mean-0'], regions['intensity_mean-1'], regions['intensity_mean-2']):
        im[:, label-1] = [r, g, b]
        
    
    im = torch.tensor(im).to(torch.uint8).reshape(3, 28, 28)
    import torchvision.transforms.functional as F
    brightness = F.adjust_brightness(im, 2).reshape(3, 784).detach().cpu().numpy()
    saturation = F.adjust_saturation(im, 2).reshape(3, 784).detach().cpu().numpy()
    contrast = F.adjust_contrast(im, 2).reshape(3, 784).detach().cpu().numpy()
    sharpness = F.adjust_sharpness(im, 2).reshape(3, 784).detach().cpu().numpy()
    posterize = F.posterize(im, 4).reshape(3, 784).detach().cpu().numpy()
    solarize = F.solarize(im, 125).reshape(3, 784).detach().cpu().numpy()
    autocontrast = F.autocontrast(im).reshape(3, 784).detach().cpu().numpy()
    equalize = F.equalize(im).reshape(3, 784).detach().cpu().numpy()
    im = im.reshape(3, 784)
    
    for label in regions['label']:
        
        im_plot[segments == label] = im[:, label-1]

    
    ax[0, 0].imshow(im_plot/255.)
    ax[0, 0].axis('off')
    ax[0, 0].set_title('Superpixel Colour Features', fontsize=15)

    for label in regions['label']:
        im_plot[segments == label] = brightness[:, label-1]
    ax[0, 1].imshow(im_plot/255)
    ax[0, 1].axis('off')
    ax[0, 1].set_title('Brightness', fontsize=15)


    for label in regions['label']:
        im_plot[segments == label] = saturation[:, label-1]
    ax[0, 2].imshow(im_plot/255)
    ax[0, 2].axis('off')
    ax[0, 2].set_title('Saturation', fontsize=15)


    for label in regions['label']:
        im_plot[segments == label] = contrast[:, label-1]
    ax[1, 0].imshow(im_plot/255)
    ax[1, 0].axis('off')
    ax[1, 0].set_title('Contrast', fontsize=15)


    for label in regions['label']:
        im_plot[segments == label] = sharpness[:, label-1]
    ax[1, 1].imshow(im_plot/255)
    ax[1, 1].axis('off')
    ax[1, 1].set_title('Sharpness', fontsize=15)


    for label in regions['label']:
        im_plot[segments == label] = posterize[:, label-1]
    ax[1, 2].imshow(im_plot/255)
    ax[1, 2].axis('off')
    ax[1, 2].set_title('Posterize', fontsize=15)


    for label in regions['label']:
        im_plot[segments == label] = solarize[:, label-1]
    ax[2, 0].imshow(im_plot/255)
    ax[2, 0].axis('off')
    ax[2, 0].set_title('Solarize', fontsize=15)


    for label in regions['label']:
        im_plot[segments == label] = autocontrast[:, label-1]
    ax[2, 1].imshow(im_plot/255)
    ax[2, 1].axis('off')
    ax[2, 1].set_title('Autocontrast', fontsize=15)


    for label in regions['label']:
        im_plot[segments == label] = equalize[:, label-1]
    ax[2, 2].imshow(im_plot/255)
    ax[2, 2].axis('off')
    ax[2, 2].set_title('Equalize', fontsize=15)

    fig.tight_layout()

    fig.savefig('/mnt/d/Figures/SuperFormer/colour_augmentation.pdf', format='pdf')
    plt.show()
    assert(0)
