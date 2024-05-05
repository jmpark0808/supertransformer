import os
from PIL import Image
import numpy as np


path = '/mnt/dragon/Datasets/YoutubeVOS/train/Annotations'
train_path = '/mnt/dragon/Datasets/YoutubeVOS/train/'

for root, subdirs, files in os.walk(path):
    for file in files:
        tag_mask = os.path.join(root.split('/')[-1], file)
        tag_image = tag_mask.replace('png', 'jpg')
        annotation_path = os.path.join(train_path, 'Annotations', tag_mask)
        image_path = os.path.join(train_path, 'JPEGImages',  tag_image)
        img = Image.open(annotation_path)
        img = np.array(img)
        unique_values = np.unique(img)
        print(unique_values)
        