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
segment_numbers = [625, 625, 625]
compactness = [10, 10, 10]
ec = [True, False, False]
sz = [True, True, False]
x_label = []
all_ious = []
plt.figure(figsize=(10,10))
for seg, c, e, s in zip(segment_numbers, compactness, ec, sz):
    ious = []
    for file in tqdm(os.listdir(dataset_images)[:1000]):
        name = file.split('.jpg')[0]
        image = os.path.join(dataset_images, name+'.jpg')
        mask = os.path.join(masks, name+'.png')

        img = Image.open(image)
        msk = Image.open(mask)
        img = img.convert('RGB').resize((300, 300))
        msk = msk.convert('L').resize((300, 300))
        img = np.array(img)
        msk = np.array(msk)
        
        # msk[msk>125] = 255
        # msk[msk<=125] = 0

        # empty_background = np.zeros_like(msk)

        # msk_boundaries = np.sum(mark_boundaries(empty_background, msk), axis=2)

        msk[msk<=125] = 0
        msk[msk>125] = 1
        

        segments = slic(img, n_segments=seg,
        compactness=c,
        max_num_iter=10,
        convert2lab=True,
        enforce_connectivity=e,
        slic_zero=s)
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
        plt_image = np.ravel(plt_image)
            

        msk = np.ravel(msk)
        y_temp = (plt_image >= 0.5).astype(np.float)
        tp = np.sum((y_temp * msk))
        # avoid prec becomes 0
        prec, recall = (tp + 1e-10) / (np.sum(y_temp) + 1e-10), (tp + 1e-10) / (np.sum(msk) + 1e-10)
        beta_square = 0.3
        f_score = (1 + beta_square) * prec * recall / (beta_square * prec + recall)
        ious.append(f_score)
    x_label.append(f'SZ {s}, EC {e}')
    all_ious.append(np.mean(ious))

# plt.plot(segment_numbers, all_ious, label=f'SZ {s}, EC {e}')
plt.bar(list(range(len(all_ious))), all_ious, align='center')



fs = 20
plt.title(f'Segmentation boundary intersection accuracy', fontsize=fs)
plt.xlabel('Segmentations', fontsize=fs)
plt.ylabel('Intersection Accuracy', fontsize=fs)
# plt.xscale('log')
plt.xticks(list(range(len(all_ious))), x_label, fontsize=fs, rotation=45)
plt.yticks(fontsize=fs)
plt.ylim((0.95, 1.0))
plt.tight_layout()
plt.savefig(f'Slic_zero.jpg')
    

