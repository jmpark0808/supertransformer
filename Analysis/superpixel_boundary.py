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
segment_numbers = [10, 15, 20, 25, 30, 40, 50, 100, 200, 300]
# segment_numbers = [100, 200, 300, 400, 500, 600, 800, 1000, 1500, 3000, 10000, 45000, 90000]
compactness = [0.1, 1, 10, 50]
d= {}
d['segment_numbers'] = segment_numbers
num_images = 3000
use_pickle = True


plt.figure(figsize=(10,10))
if use_pickle:
    d = open("segments_plot_data.pkl",'rb')
    d = pickle.load(d)
    for compact in tqdm(compactness):
        all_ious = d[compact] 
        plt.plot(np.power(np.array(segment_numbers), 2), all_ious, label=f'{compact}')
        plt.scatter(np.power(np.array(segment_numbers), 2), all_ious)
        for i, j in zip(np.power(np.array(segment_numbers), 2), all_ious):
            plt.text(i, j+0.002, '{}'.format(i))

    all_ious = d['ious'] 
    plt.plot(np.power(np.array(segment_numbers), 2), all_ious, label='Resize')
    plt.scatter(np.power(np.array(segment_numbers), 2), all_ious)
    for i, j in zip(np.power(np.array(segment_numbers), 2), all_ious):
        plt.text(i, j+0.002, '{}'.format(i))

else:
    for compact in tqdm(compactness):
        all_ious = []
        for seg in segment_numbers:
            IoUs = []
            for file in tqdm(os.listdir(dataset_images)[:num_images]):
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
                
                num_seg = seg*seg
                segments = slic(img, n_segments=num_seg,
                compactness=compact,
                max_num_iter=10,
                convert2lab=True,
                enforce_connectivity=False,
                slic_zero=False)

                # segments = slic(image=img, n_segments=seg, compactness=compact, min_size_factor=0.5, max_num_iter=3, enforce_connectivity=False)
                # segments = slic.iterate(img)

                # superpixel_boundaries = np.sum(mark_boundaries(empty_background, segments), axis=2)

                # iou = np.sum(np.logical_and((msk_boundaries == 2),(superpixel_boundaries == 2)))/np.sum(msk_boundaries>0)
                regions = regionprops_table(segments, properties=('label', 'coords', ))
                try:
                    max(regions['label'])
                except:
                    plt.imshow(img)
                    plt.show()
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
                IoUs.append(f_score)

            all_ious.append(np.mean(IoUs))
        d[compact] = all_ious
        plt.plot(np.power(np.array(segment_numbers), 2), all_ious, label=f'{compact}')
        plt.scatter(np.power(np.array(segment_numbers), 2), all_ious)
        for i, j in zip(np.power(np.array(segment_numbers), 2), all_ious):
            plt.text(i, j+0.002, '{}'.format(i))

    all_ious = []
    for resolution in segment_numbers:
        IoUs = []
        for file in tqdm(os.listdir(dataset_images)[:num_images]):
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
    plt.plot(np.power(np.array(segment_numbers), 2), all_ious, label='Resize')
    plt.scatter(np.power(np.array(segment_numbers), 2), all_ious)
    for i, j in zip(np.power(np.array(segment_numbers), 2), all_ious):
        plt.text(i, j+0.002, '{}'.format(i))

with open('segments_plot_data.pkl', 'wb') as f:
    pickle.dump(d, f)
fs = 20
plt.title(f'Segmentation boundary intersection accuracy', fontsize=fs)
plt.xlabel('Segmentations', fontsize=fs)
plt.ylabel('Intersection Accuracy', fontsize=fs)
plt.xscale('log')
plt.xticks(fontsize=fs, rotation=45)
plt.yticks(fontsize=fs)
plt.legend(loc="lower right", fontsize=fs, title='Compactness', title_fontsize=fs)
for vertical in np.power(np.array(segment_numbers), 2):
    plt.axvline(x=vertical, linestyle='--')
plt.savefig(f'compactness.jpg')
    

