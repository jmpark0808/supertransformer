import os
import matplotlib.pyplot as plt
import cv2
from skimage import io
from skimage.segmentation import mark_boundaries, slic
from skimage.measure import regionprops_table
import numpy as np
from PIL import Image
from tqdm import tqdm
import pickle

dataset_images = '/mnt/hdd/Datasets/DUTS/DUTS-TR/Image'
masks = '/mnt/hdd/Datasets/DUTS/DUTS-TR/Mask'
image_resolutions = [300, 256, 128, 64, 32, 16, 8]

d= {}
d['image_resolutions'] = image_resolutions
plt.figure(figsize=(10,10))

all_ious = []
for resolution in image_resolutions:
    IoUs = []
    for file in tqdm(os.listdir(dataset_images)[:5000]):
        name = file.split('.jpg')[0]
        image = os.path.join(dataset_images, name+'.jpg')
        mask = os.path.join(masks, name+'.png')

        msk = Image.open(mask)
        msk_resize = Image.open(mask)
        msk = msk.convert('L').resize((300, 300))
        msk_resize = msk_resize.convert('L').resize((resolution, resolution))

        msk = np.array(msk)
        msk_resize = np.array(msk_resize)
        

        msk[msk<=125] = 0
        msk[msk>125] = 1

        msk_resize[msk_resize<=125] = 0
        msk_resize[msk_resize>125] = 1
                   
        plt_image = cv2.resize(msk_resize, (300, 300))
        plt_image = np.ravel(plt_image)
        msk = np.ravel(msk)
        y_temp = (plt_image >= 0.5).astype(np.float)
        tp = np.sum((y_temp * msk))
        # avoid prec becomes 0
        prec, recall = (tp + 1e-10) / (np.sum(y_temp) + 1e-10), (tp + 1e-10) / (np.sum(msk) + 1e-10)
        beta_square = 0.3
        f_score = (1 + beta_square) * prec * recall / (beta_square * prec + recall)
        IoUs.append(f_score)

    all_ious.append(np.mean(IoUs))

d['ious'] = all_ious
plt.plot(image_resolutions, all_ious)
plt.scatter(image_resolutions, all_ious)
for i, j in zip(image_resolutions, all_ious):
    plt.text(i, j+0.002, '{}'.format(i))

with open('resizing_plot_data.pkl', 'wb') as f:
    pickle.dump(d, f)
fs = 20
plt.title(f'Resize boundary intersection accuracy', fontsize=fs)
plt.xlabel('Image resolution', fontsize=fs)
plt.ylabel('Intersection Accuracy', fontsize=fs)
plt.xscale('log')
plt.xticks(fontsize=fs, rotation=45)
plt.yticks(fontsize=fs)
# plt.axvline(x=image_resolutions)
plt.savefig(f'resize_accuracy.jpg')
    

