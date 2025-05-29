

import matplotlib.pyplot as plt
from PIL import Image
import os
import numpy as np
from skimage.segmentation import slic, mark_boundaries
from skimage.measure import regionprops_table

image_1 = '/home/eddie/Datasets/DUTS/DUTS-TE/Image/ILSVRC2012_test_00000105.jpg'





img_1 = Image.open(image_1)





img_1 = img_1.convert('RGB').resize((448, 448))




img_1 = np.array(img_1)



num_seg = 784

segments = slic(img_1, n_segments=num_seg,
compactness=10,
max_num_iter=10,
convert2lab=True,
enforce_connectivity=True,
slic_zero=True)

fig, ax = plt.subplots(1, 2, figsize=(10,5))
ax[0].imshow(img_1)
ax[1].imshow(mark_boundaries(img_1, segments, (1,1,1)))
ax[0].axis('off')
ax[1].axis('off')
plt.savefig('/mnt/hdd/Figures/SuperFormer/intro.pdf', format='pdf')
plt.show()