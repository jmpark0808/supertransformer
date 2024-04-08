import os
import numpy as np

from fast_slic import Slic
from PIL import Image
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries, slic
from skimage.measure import regionprops, regionprops_table

root_dir = '/mnt/nas/PhD/Datasets/DUTS/DUTS-TR/DUTS-TR-Image'
xs = []
ys = []
count_up_to = 5
for ind, path in enumerate(os.listdir(root_dir)):
    if count_up_to == ind:
        break
    with Image.open(os.path.join(root_dir, path)) as f:
        f = f.resize((300, 300))
        image = np.array(f)
    # import cv2; image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)   # You can convert the image to CIELAB space if you need.
    segments = slic(image, n_segments=580,
      compactness=10.0,
      max_num_iter=10,
      convert2lab=True,
      enforce_connectivity=True,
      slic_zero=True)
    
    
    regions = regionprops(segments, intensity_image=image)
    regions_table = regionprops_table(segments, intensity_image=image, properties=('label', 'centroid', 'area', 'intensity_mean', 'extent', 'coords', 'eccentricity'))
    
    print(regions_table['label'], len(regions_table['label']))
    
    # for props in regions:
    #     if props.label == 10:
    #         cx, cy = props.centroid  # centroid coordinates
    #         size = props.area
    #         means = props.intensity_mean
    #         ecc = props.eccentricity
    #         extent = props.extent
    #         coords = props.coords
    #         print(image[coords[:, 0], coords[:, 1]])
    #         print(cx, cy, size, means, ecc, extent)
    #         assert(0)
            

        


