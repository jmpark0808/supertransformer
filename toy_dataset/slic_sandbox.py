import numpy as np

# Much faster than the standard class
from fast_slic.avx2 import SlicAvx2
from PIL import Image

with Image.open("/mnt/hdd/Datasets/DUTS/DUTS-TE/Image/ILSVRC2012_test_00000003.jpg") as f:
   image = np.array(f)
# import cv2; image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)   # You can convert the image to CIELAB space if you need.
slic = SlicAvx2(num_components=600, compactness=10, min_size_factor=0)
assignment = slic.iterate(image) # Cluster Map
