from PIL import Image
import numpy as np
import os

labels= []
for mask in os.listdir('/home/eddie/Datasets/COCOStuff/Val/Mask/'):
    mask_img = Image.open(os.path.join('/home/eddie/Datasets/COCOStuff/Val/Mask/', mask))
    mask_np = np.array(mask_img)
    labels.extend(np.unique(mask_np))
  
print(np.unique(labels))