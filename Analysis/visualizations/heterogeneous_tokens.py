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

image_path = '/mnt/hdd/Datasets/DUTS/DUTS-TR/Image/'
ind = 0
for i in os.listdir(image_path):
    img = Image.open(os.path.join(image_path, i)).convert('RGB')
    img = img.resize((300, 300))

    img = np.array(img)

    fig, ax = plt.subplots(1, 2, figsize=(20, 10))


    # slic = SlicAvx2(num_components=625, compactness=50, min_size_factor=0.)
    # segments = slic.iterate(img, max_iter=0)+1

    segments = slic(img, n_segments=400,
                compactness=30,
                max_num_iter=10,
                convert2lab=True,
                enforce_connectivity=False,
                slic_zero=False)
    
    regions = regionprops_table(segments, img, properties=('label', 'centroid', 'intensity_mean'))
    # print(np.arange(0, 400).reshape(20, 20))
    
    out = color.label2rgb(segments, img, kind='avg', bg_label=0)
    
    indices = np.array([100, 101, 102, 120, 121, 122, 140, 141, 142])+10
    colors = [[0, 0, 255], [0, 0, 255], [0, 0, 255], [0, 0, 255], [0, 255, 0], [0, 0, 255], [0, 0, 255], [0, 0, 255], [0, 0, 255]]
    for idx, c in zip(indices, colors):
        mask = segments == idx
        out[mask] = c
    out = segmentation.mark_boundaries(out, segments, (0, 0, 0))
    ax[0].imshow(out)
    ax[0].set_title('SLIC/Superpixel', fontsize=20)
    ax[0].axis('off')

    
    img[90:105, 150:165] = [0, 255, 0]
    img[90:105, 135:150] = [0, 0, 255]
    img[90:105, 165:180] = [0, 0, 255]
    img[75:90, 150:165] = [0, 0, 255]
    img[105:120, 150:165] = [0, 0, 255]
    img[75:90, 135:150] = [0, 0, 255]
    img[105:120, 135:150] = [0, 0, 255]
    img[75:90, 165:180] = [0, 0, 255]
    img[105:120, 165:180] = [0, 0, 255]


    ax[1].imshow(img)
    # ax[1].plot(list(range(300)))
    ax[1].set_title('Vision Transformer', fontsize=20)
    ax[1].axis('off')

    for i in range(0, 300, 15):
        if i != 0:
            ax[1].vlines(x=i, ymin=0, ymax=299, lw=2, color='r')
            ax[1].hlines(y=i, xmin=0, xmax=299, lw=2, color='r')

        
    fig.subplots_adjust(top=1.05)

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])    
    fig.savefig(f'/home/eddie/Downloads/gif/{ind}.jpg')
    ind += 1
    if ind == 15: 
        break
   
