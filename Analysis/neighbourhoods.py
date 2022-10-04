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
CHUNK = 10
NUM_CHUNK = 360//CHUNK

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
            radii_max[ind] = normalized_coords[np.argmax(rho[(degree<=phi) & (phi<degree+chunk)])]
            radii_min[ind] = normalized_coords[np.argmin(rho[(degree<=phi) & (phi<degree+chunk)])]
        except: 
            pass
        

    return np.concatenate((radii_max, radii_min), axis=0)


data_dir = '/mnt/hdd/Datasets/DUTS/DUTS-TR/Image/'
all_distances = []
for file in os.listdir(data_dir)[:1000]:
    img = Image.open(os.path.join(data_dir, file))
    img = img.convert('RGB')
    img = img.resize((300, 300), resample=Image.BILINEAR)

    img_np = np.array(img).astype(np.float32)/255.




    img_size = img_np.shape[1]

    start = time.time()
    segments = slic(img, n_segments=625,
        compactness=10.0,
        max_num_iter=10,
        convert2lab=True,
        enforce_connectivity=False,
        slic_zero=False)
    end = time.time()
    print('SLIC time', end-start)
    # segments = slic.iterate(img_np)

    vs_right = np.vstack([segments[:,:-1].ravel(), segments[:,1:].ravel()])
    vs_below = np.vstack([segments[:-1,:].ravel(), segments[1:,:].ravel()])
    vs_diagonal_r = np.vstack([segments[:-1,:-1].ravel(), segments[1:,1:].ravel()])
    vs_diagonal_l = np.vstack([segments[1:,:-1].ravel(), segments[:-1,1:].ravel()])
    bneighbors = np.unique(np.hstack([vs_right, vs_below, vs_diagonal_r, vs_diagonal_l]), axis=1)
    # bneighbors = np.unique(np.hstack([vs_right, vs_below]), axis=1)

    regions = regionprops_table(segments, intensity_image=img_np, properties=('label', 'centroid', 'bbox', 'area', 'intensity_mean', 'extent', 'coords', 'eccentricity'), extra_properties=[polarize])
    seq_len = max(regions['label'])

    print(regions.keys())
    print(len(regions['area']))
    assert(0)

    features = np.zeros([seq_len, 8])
    seq_mask = np.zeros([seq_len])
    label = regions['label']

    features[label-1, 0] = regions['centroid-0']
    features[label-1, 1] = regions['centroid-1']
    features[label-1, 2] = regions['area'] / (img_size**2)
    features[label-1, 3] = regions['intensity_mean-0']/255.
    features[label-1, 4] = regions['intensity_mean-1']/255.
    features[label-1, 5] = regions['intensity_mean-2']/255.
    features[label-1, 6] = regions['extent']
    features[label-1, 7] = regions['eccentricity']


    neighbor_array = np.zeros([seq_len, seq_len])
    neighbor_array[bneighbors[0]-1, bneighbors[1]-1] = 1
    neighbor_array[bneighbors[1]-1, bneighbors[0]-1] = 1

    distances = euclidean_distances(features[:, :2], features[:, :2])
    all_distances.append(np.mean(distances[np.nonzero(distances*neighbor_array)]))
    print(np.mean(distances[np.nonzero(distances*neighbor_array)]))
    # Try dilation
    # neighbor_array = np.linalg.matrix_power(neighbor_array, 1).astype(bool).astype(int) - np.linalg.matrix_power(neighbor_array, 0).astype(bool).astype(int) 
    # + np.eye(seq_len).astype(bool).astype(int)
    # # neighbor_array = neighbor_array.astype(bool).astype(int)

    # random_sp = np.random.randint(0, 599)

    # segments_ids = np.unique(segments)

    # # centers
    # centers = np.array([np.mean(np.nonzero(segments==i),axis=1) for i in segments_ids])

    # fig = plt.figure(figsize=(10,10))
    # ax = fig.add_subplot(111)
    # plt.imshow(mark_boundaries(img_np, segments))
    # plt.scatter(centers[:,1],centers[:,0], c='blue', s=30)
    # for ind, (x, y) in enumerate(zip(features[:, 1], features[:, 0])):
    #     plt.text(x, y, str(regions['label'][ind]))

    # plt.scatter(features[:, 1], features[:, 0], c='red', s=30)

    # for neighbours in np.argwhere(neighbor_array[random_sp]==1):
    #     plt.scatter(features[neighbours, 1], features[neighbours, 0], s=40, c='blue')

    # plt.scatter(features[random_sp, 1], features[random_sp, 0], s=40, c='red')


    '''
    Draws line between centroids
    '''
    # for i in range(bneighbors.shape[1]):
    #     y0,x0 = centers[bneighbors[0,i]-1]
    #     y1,x1 = centers[bneighbors[1,i]-1]

    #     l = Line2D([x0,x1],[y0,y1], alpha=0.5)
    #     ax.add_line(l)

    # plt.show()


    # assert(0)

    
print(np.mean(all_distances))