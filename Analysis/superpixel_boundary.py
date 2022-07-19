import os
import matplotlib.pyplot as plt
import cv2
from skimage import io
from skimage.segmentation import mark_boundaries, slic
import numpy as np
from PIL import Image
from tqdm import tqdm


dataset_images = '/mnt/hdd/Datasets/DUTS/TR/Image'
masks = '/mnt/hdd/Datasets/DUTS/TR/Mask'
segment_numbers = [90000, 45000, 15000, 5000, 2500, 1500, 1000, 800, 600, 500, 400, 300, 200, 100]
compactness = [0.1, 10]
plt.figure(figsize=(10,10))
for compact in compactness:
    all_ious = []
    for seg in segment_numbers:
        IoUs = []
        for file in tqdm(os.listdir(dataset_images)[:1000]):
            name = file.split('.jpg')[0]
            image = os.path.join(dataset_images, name+'.jpg')
            mask = os.path.join(masks, name+'.png')

            img = Image.open(image)
            msk = Image.open(mask)
            img = img.convert('RGB')
            msk = msk.convert('L')
            img = np.array(img)
            msk = np.array(msk)
            
            msk[msk>125] = 255
            msk[msk<=125] = 0

            empty_background = np.zeros_like(msk)

            msk_boundaries = np.sum(mark_boundaries(empty_background, msk), axis=2)

            
            segments = slic(image=img, n_segments=seg, compactness=compact, min_size_factor=0, max_num_iter=3, enforce_connectivity=False)
            # segments = slic.iterate(img)

            superpixel_boundaries = np.sum(mark_boundaries(empty_background, segments), axis=2)

            iou = np.sum(np.logical_and((msk_boundaries == 2),(superpixel_boundaries == 2)))/np.sum(msk_boundaries>0)  
            IoUs.append(iou)

        all_ious.append(np.mean(IoUs))
    plt.plot(segment_numbers, all_ious, label=f'{compact}')
    plt.scatter(segment_numbers, all_ious)
    for i, j in zip(segment_numbers, all_ious):
        plt.text(i, j+0.01, '{}'.format(i))
fs = 20
plt.title(f'Segmentation boundary intersection accuracy', fontsize=fs)
plt.xlabel('Segmentations', fontsize=fs)
plt.ylabel('Intersection Accuracy', fontsize=fs)
plt.xscale('log')
plt.xticks(fontsize=fs, rotation=45)
plt.yticks(fontsize=fs)
plt.legend(loc="lower right", fontsize=fs, title='Compactness', title_fontsize=fs)
plt.savefig(f'compactness.jpg')
    

