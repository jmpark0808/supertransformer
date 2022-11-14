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
segment_numbers = [100, 1000, 10000]

fig, ax = plt.subplots(2, 2)
total_count = len(os.listdir(dataset_images))
random_ind = np.random.randint(0, total_count)


file = os.listdir(dataset_images)[random_ind]
name = file.split('.jpg')[0]
image = os.path.join(dataset_images, name+'.jpg')
mask = os.path.join(masks, name+'.png')

img = Image.open(image)
msk = Image.open(mask)
img = img.convert('RGB').resize((300, 300))
msk = msk.convert('L').resize((300, 300))
img = np.array(img)
msk = np.array(msk)

msk[msk<=125] = 0
msk[msk>125] = 1

for i, seg in enumerate(segment_numbers):
    
    

    segments = slic(img, n_segments=seg,
    compactness=10,
    max_num_iter=10,
    convert2lab=True,
    enforce_connectivity=False,
    slic_zero=False)
    # segments = slic(image=img, n_segments=seg, compactness=compact, min_size_factor=0.5, max_num_iter=3, enforce_connectivity=False)
    # segments = slic.iterate(img)

    # superpixel_boundaries = np.sum(mark_boundaries(empty_background, segments), axis=2)

    # iou = np.sum(np.logical_and((msk_boundaries == 2),(superpixel_boundaries == 2)))/np.sum(msk_boundaries>0)
    regions = regionprops_table(segments, properties=('label', 'coords', ))
    seq_mask = np.zeros([max(regions['label'])])
    # assert len(regions['label']) == max(regions['label']), 'Wrong number of labels'

    for ind, coord in zip(regions['label'], regions['coords']):
        seq_mask[ind-1] = np.sum(msk[coord[:, 0], coord[:, 1]])/len(coord[:, 0])

    plt_image = seq_mask[segments-1].reshape([img.shape[0], img.shape[1]])
    # plt_image = np.ravel(plt_image)

    ax[i//2, i%2].imshow(plt_image, cmap='gray')
    ax[i//2, i%2].set_title(f"{np.max(regions['label'])} segmentations")
    ax[i//2, i%2].axis('off')

ax[1, 1].imshow(img)
ax[1, 1].set_title("Raw Image")
ax[1, 1].axis('off')

plt.savefig('plot_compactness_mask.jpg')
    