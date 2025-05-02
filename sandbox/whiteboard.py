from PIL import Image
import numpy as np
import os

labels= []
for features in os.listdir('/home/eddie/Datasets/DUTS/DUTS-TE/SPFFFT'):
    if 'features' in features:
        feature = np.load(os.path.join('/home/eddie/Datasets/DUTS/DUTS-TE/SPFFFT', features))

        
  
        print(np.sum(feature[:, -18]))