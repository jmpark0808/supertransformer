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

data_dir = '/mnt/hdd/Datasets/DUTS/DUTS-TR/Image/'
all_distances = []
heights = []
widths = []
for file in tqdm(os.listdir(data_dir)):
    img = Image.open(os.path.join(data_dir, file))
    img = img.convert('RGB')
    img = img.resize((1000, 1000), resample=Image.BILINEAR)

    img_np = np.array(img).astype(np.float32)/255.

    num_seg = 256


    img_size = img_np.shape[1]

    start = time.time()
    segments = slic(img, n_segments=num_seg,
        compactness=10.0,
        max_num_iter=10,
        convert2lab=True,
        enforce_connectivity=True,
        slic_zero=True)
    vs_right = np.vstack([segments[:,:-1].ravel(), segments[:,1:].ravel()])
    vs_below = np.vstack([segments[:-1,:].ravel(), segments[1:,:].ravel()])
    vs_diagonal_r = np.vstack([segments[:-1,:-1].ravel(), segments[1:,1:].ravel()])
    vs_diagonal_l = np.vstack([segments[1:,:-1].ravel(), segments[:-1,1:].ravel()])
    bneighbors = np.unique(np.hstack([vs_right, vs_below, vs_diagonal_r, vs_diagonal_l]), axis=1)
    end = time.time()

    regions = regionprops_table(segments, intensity_image=img_np, properties=('label', 'centroid', 'area', 'intensity_mean',
                                                                                'extent', 'coords', 'eccentricity'), extra_properties=[image_stdev, hist])#, polarize])
                
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
    features[label-1, 8] = regions['image_stdev-0']/255.
    features[label-1, 9] = regions['image_stdev-1']/255.
    features[label-1, 10] = regions['image_stdev-2']/255.


    neighbor_array = np.zeros([num_seg, num_seg])
    # eye = np.eye(self.num_seg)
    neighbor_array[bneighbors[0]-1, bneighbors[1]-1] = 1
    neighbor_array[bneighbors[1]-1, bneighbors[0]-1] = 1
    # neighbor_array -= eye


    # A = neighbor_array.astype(float)
    # N = sp.diags(np.sum(A, axis=0)** -0.5, dtype=float)
    # L = eye - N * A * N

    # # Eigenvectors with numpy
    # EigVal, EigVec = np.linalg.eig(L)
    # idx = EigVal.argsort() # increasing order
    # EigVal, EigVec = EigVal[idx], np.real(EigVec[:,idx])
    # pos_enc = torch.from_numpy(EigVec[:,1:POS_EMBEDDING+1]).float() 

    histogram_r = np.zeros([num_seg, BINS])
    histogram_g = np.zeros([num_seg, BINS])
    histogram_b = np.zeros([num_seg, BINS])
    for i in range(BINS):
        histogram_r[label-1, i] = regions[f'hist-{i}-0']
        histogram_g[label-1, i] = regions[f'hist-{i}-1']
        histogram_b[label-1, i] = regions[f'hist-{i}-2']

    histogram_r = histogram_r/np.sum(histogram_r, axis=1, keepdims=True)
    histogram_g = histogram_g/np.sum(histogram_g, axis=1, keepdims=True)
    histogram_b = histogram_b/np.sum(histogram_b, axis=1, keepdims=True)
    
    histogram_r_sq = 1-pdist(histogram_r, lambda u, v: np.sqrt(u*v).sum())
    histogram_g_sq = 1-pdist(histogram_g, lambda u, v: np.sqrt(u*v).sum())
    histogram_b_sq = 1-pdist(histogram_b, lambda u, v: np.sqrt(u*v).sum())

    # spatial_distances = euclidean_distances(features[:, :2], features[:, :2])/np.sqrt(300**2+300**2)
    spatial_distances_x = (features[:, 0:1] - features[:, 0:1].T)/300.
    spatial_distances_y = (features[:, 1:2] - features[:, 1:2].T)/300.
    # ind = np.argsort(distances, axis=1)
    # neighbor_array = ind <= NUM_NEIGHBOURS
    # neighbor_array = np.zeros([self.num_seg, self.num_seg])
    
    edge_features = np.stack((spatial_distances_x, spatial_distances_y, squareform(histogram_r_sq), squareform(histogram_g_sq), squareform(histogram_b_sq)), axis=2)

    # end = time.time()
    print(end-start)
    # for i in range(NUM_CHUNK*2):
    #     features[label-1, 11+i] = regions[f'polarize-{i}-0']
    #     features[label-1, 11+NUM_CHUNK*2+i] = regions[f'polarize-{i}-1']


    # distances = euclidean_distances(features[:, :2], features[:, :2])
    # ind = np.argsort(distances, axis=1)
    # ranged_ind = np.array(range(self.num_seg))
    # ranged_ind = np.tile(ranged_ind, (self.num_seg, 1)) 
    # ind = np.take_along_axis(ranged_ind, ind, axis=1)
    # ind = ind[:, :NUM_NEIGHBOURS]
    # neighbor_array = distances <= 30

    # distances = euclidean_distances(features[:, :2], features[:, :2])
    # all_distances.append(np.mean(distances[np.nonzero(distances*neighbor_array)]))
    # print(np.mean(distances[np.nonzero(distances*neighbor_array)]))

    # cen = features[:, :2] # 625, 2
    # cen = np.repeat(cen[:, None, :], NUM_CHUNK*2, axis=1)
    # pos_x = features[:, 11:11+NUM_CHUNK*2] # 625, 72
    # pos_y = features[:, 11+NUM_CHUNK*2:] # 625, 72
    # pos = np.concatenate((pos_x[:, :, None], pos_y[:, :, None]), axis=2)+cen
    # relative_distances = pos[None, :, :, :]-cen[:, None, :, :] # 625, 625, 72, 2
   

    # neighbor_index = 125
    # centroid = features[neighbor_index, :2]
    # all_sp = relative_distances[neighbor_index, np.squeeze(np.argwhere(neighbor_array[neighbor_index, :] == 1)), :, :]
    # all_sp = pos[np.squeeze(np.argwhere(neighbor_array[neighbor_index, :] == 1)), :, :]
    
    # for sp in all_sp:
    #     minimums = sp[36:, :]
    #     maximums = sp[:36, :]
    #     plt.scatter(maximums[:, 1], maximums[:, 0])
    #     plt.scatter(minimums[:, 1], minimums[:, 0], c='blue')
    # # plt.scatter(0, 0, c='orange', s=30)
    # plt.show()
    # assert(0)
        



    # Try dilation
    dilation = 2
    neighbor_array = np.linalg.matrix_power(neighbor_array, dilation).astype(bool).astype(int)
    # neighbor_array = neighbor_array.astype(bool).astype(int)

    random_sp = 100

    segments_ids = np.unique(segments)

    # centers
    centers = np.array([np.mean(np.nonzero(segments==i),axis=1) for i in segments_ids])

    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)
    plt.imshow(mark_boundaries(img_np, segments))
    # plt.scatter(centers[:,1],centers[:,0], c='blue', s=30)
    # for ind, (x, y) in enumerate(zip(features[:, 1], features[:, 0])):
    #     plt.text(x, y, str(regions['label'][ind]))

    plt.scatter(features[:, 1], features[:, 0], c='red', s=40)

    for neighbours in np.argwhere(neighbor_array[random_sp]==1):
        plt.scatter(features[neighbours, 1], features[neighbours, 0], s=40, c='blue')

    plt.scatter(features[random_sp, 1], features[random_sp, 0], c='green', s=40)

    # plt.scatter(features[random_sp, 1], features[random_sp, 0], s=40, c='red')


    '''
    Draws line between centroids
    '''
    # for i in range(bneighbors.shape[1]):
    #     y0,x0 = centers[bneighbors[0,i]-1]
    #     y1,x1 = centers[bneighbors[1,i]-1]

    #     l = Line2D([x0,x1],[y0,y1], alpha=0.5)
    #     ax.add_line(l)

    plt.show()


    # assert(0)

    
# print(np.min(heights), np.max(heights))
# print(np.min(widths), np.max(widths))