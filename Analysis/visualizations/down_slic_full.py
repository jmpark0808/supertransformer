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
for i in os.listdir(image_path):
    img = Image.open(os.path.join(image_path, i)).convert('RGB')
    img_resize = img.resize((25, 25))
    img_resize = img_resize.resize((300, 300))
    img = img.resize((300, 300))

    img = np.array(img)

    fig, ax = plt.subplots(1, 3, figsize=(20, 10))


    # slic = SlicAvx2(num_components=625, compactness=50, min_size_factor=0.)
    # segments = slic.iterate(img, max_iter=0)+1

    segments = slic(img, n_segments=625,
                compactness=10,
                max_num_iter=10,
                convert2lab=True,
                enforce_connectivity=True,
                slic_zero=False)
    
    regions = regionprops_table(segments, img, properties=('label', 'centroid', 'intensity_mean'))
  

    out = color.label2rgb(segments, img, kind='avg', bg_label=0)
    out = segmentation.mark_boundaries(out, segments, (0, 0, 0))
    ax[0].imshow(out)
    ax[0].set_title('SLIC/Superpixel', fontsize=20)
    ax[0].axis('off')

    img_resize = np.array(img_resize)
    ax[1].imshow(img_resize)
    ax[1].set_title('Downsampling', fontsize=20)
    ax[1].axis('off')

    ax[2].imshow(img)
    ax[2].set_title('Full Image', fontsize=20)
    ax[2].axis('off')
    
    plt.show()
