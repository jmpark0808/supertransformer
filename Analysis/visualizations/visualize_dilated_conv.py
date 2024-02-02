import numpy as np
from PIL import Image
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
from skimage import data, io, segmentation, color
from skimage import graph
from skimage.morphology import disk
from skimage.segmentation import watershed
from skimage import data
from skimage.filters import rank
from skimage.util import img_as_ubyte
from skimage.segmentation import mark_boundaries, slic
from skimage.measure import regionprops_table
import os
from fast_slic.avx2 import SlicAvx2
from matplotlib.lines import Line2D

image_path = '/mnt/dragon/Datasets/DUTS/DUTS-TR/Image/'
for i in os.listdir(image_path):
    img = Image.open(os.path.join(image_path, i)).convert('RGB')
    img = img.resize((300, 300))
    img = np.array(img)

    
    # slic = SlicAvx2(num_components=625, compactness=50, min_size_factor=0.)
    # segments = slic.iterate(img, max_iter=0)+1

    segments = slic(img, n_segments=625,
                compactness=10,
                max_num_iter=10,
                convert2lab=True,
                enforce_connectivity=False,
                slic_zero=False)
    
    vs_right = np.vstack([segments[:,:-1].ravel(), segments[:,1:].ravel()])
    vs_below = np.vstack([segments[:-1,:].ravel(), segments[1:,:].ravel()])
    vs_diagonal_r = np.vstack([segments[:-1,:-1].ravel(), segments[1:,1:].ravel()])
    vs_diagonal_l = np.vstack([segments[1:,:-1].ravel(), segments[:-1,1:].ravel()])

    bneighbors, counts = np.unique(np.hstack([vs_right, vs_below, vs_diagonal_r, vs_diagonal_l]), axis=1, return_counts=True)
    neighbor_array = np.zeros([625, 625])
    neighbor_array[bneighbors[0]-1, bneighbors[1]-1] = 1
    neighbor_array[bneighbors[1]-1, bneighbors[0]-1] = 1

    # neighbor_array = (np.linalg.matrix_power(neighbor_array, 7).astype(bool).astype(int) - \
    #                         np.linalg.matrix_power(neighbor_array, 7-1).astype(bool).astype(int) + \
    #                         neighbor_array).astype(bool).astype(int)

    grid = np.arange(625).reshape([25, 25])
    midpoint_indices = []
    for row in range(4):
        for column in range(4):
            midpoint_indices.append(grid[row*5+4, column*5+4])

    neighbor_array[midpoint_indices, :] = 1
    neighbor_array[:, midpoint_indices] = 1



    out = color.label2rgb(segments, img, kind='avg', bg_label=0)
    out = segmentation.mark_boundaries(out, segments, (0, 0, 0))
    for i in range(625):
        fig1, ax1 = plt.subplots()
        out_ = segmentation.mark_boundaries(out, segments == (i+1), (255, 0, 0), mode='inner')
        for neighbor in np.argwhere(neighbor_array[i, :] == 1):
            if neighbor != i:
                
                ind = np.argwhere(segmentation.find_boundaries(segments-1 == neighbor, mode='inner'))
                # plt.imshow((segmentation.find_boundaries(segments-1 == neighbor)).astype(np.int32), cmap='gray')
                # plt.show()
                

                out_[ind[:, 0], ind[:, 1]] = [0, 255, 0]
            
        ax1.imshow(out_)
        ax1.set_title(f'Global + Local Aggregation {i}')
        fig1.savefig(f'/home/eddie/Downloads/gif/{i}')
        
    

    assert(0)

