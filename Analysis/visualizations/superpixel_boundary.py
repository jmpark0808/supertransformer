import os
import matplotlib.pyplot as plt
import cv2
from skimage import io
from skimage.segmentation import mark_boundaries, slic, quickshift
from skimage.measure import regionprops_table
import numpy as np
from PIL import Image
from tqdm import tqdm
import pickle
import time
from fast_slic.avx2 import SlicAvx2
import torch
from torch_geometric.utils import scatter
import torch.nn.functional as F
from torch_scatter import scatter_std
from util.util import S_object, S_region, eval_e

dataset_images = '/home/eddie/Datasets/DUTS/DUTS-TR/Image'
masks = '/home/eddie/Datasets/DUTS/DUTS-TR/Mask'
# segment_numbers = [224, 448]# , 300
# segment_numbers = [100, 200, 300, 400, 500, 600, 800, 1000, 1500, 3000, 10000, 45000, 90000]
segment_numbers = [3136]
compactness = [10]
d= {}
d['segment_numbers'] = segment_numbers
num_images = 3000
use_pickle = False


fig, ax = plt.subplots(1, 2, figsize=(10, 10))

for compact in tqdm(compactness):
    all_ious = []
    all_maes = []
    for seg in segment_numbers:
        IoUs = []
        maes = []
        H = []
        W = []

        e_measure_scores = torch.zeros(255).cuda()
        s_measure_q = 0.0
        
        for file in tqdm(os.listdir(dataset_images)[:num_images]):
            name = file.split('.jpg')[0]
            image = os.path.join(dataset_images, name+'.jpg')
            mask = os.path.join(masks, name+'.png')

            img = Image.open(image)
            msk = Image.open(mask)
            img = img.convert('RGB').resize((448, 448))
            msk = msk.convert('L').resize((448, 448))
            img = np.array(img)
            msk = np.array(msk)
            
            H.append(img.shape[0])
            W.append(img.shape[1])
            
            msk[msk>125] = 255
            msk[msk<=125] = 0

            empty_background = np.zeros_like(msk)

            msk_boundaries = np.sum(mark_boundaries(empty_background, msk), axis=2)

            msk[msk<=125] = 0
            msk[msk>125] = 1
            
            
            start = time.time()
            segments = slic(img, n_segments=seg,
            compactness=compact,
            max_num_iter=10,
            convert2lab=True,
            enforce_connectivity=False,
            slic_zero=True)
            
            
            # segments = quickshift(img, kernel_size=3, max_dist=6, ratio=0.5)
            # slic = SlicAvx2(num_components=num_seg, compactness=compact, min_size_factor=0.)
            # segments = slic.iterate(img)

            end = time.time()
            ms = end-start
            # print(ms)
            
            

            # segments = slic(image=img, n_segments=seg,
            #                  compactness=compact,
            #                    min_size_factor=0.5, max_num_iter=10,
            #                      enforce_connectivity=False)
            # segments = slic.iterate(img)

            # superpixel_boundaries = np.sum(mark_boundaries(empty_background, segments), axis=2)

            # iou = np.sum(np.logical_and((msk_boundaries == 2),(superpixel_boundaries == 2)))/np.sum(msk_boundaries>0)
            regions = regionprops_table(segments, img, properties=('label', 'centroid', 'area', 'intensity_mean',
                                                                                 'coords',))
            end = time.time()
            ms = end-start
            
            try:
                max(regions['label'])
            except:
                plt.imshow(img)
                plt.show()
            seq_mask = np.zeros([max(regions['label'])])
            # assert len(regions['label']) == max(regions['label']), 'Wrong number of labels'

            for ind, coord in zip(regions['label'], regions['coords']):
                seq_mask[ind-1] = 1 if np.sum(msk[coord[:, 0], coord[:, 1]])/len(coord[:, 0]) >= 0.5 else 0

            plt_image = seq_mask[segments-1].reshape([img.shape[0], img.shape[1]])
            
            pred = torch.tensor(plt_image, device='cuda')
            gt = torch.tensor(msk, device='cuda').float()

            e_measure_scores += eval_e(pred, gt, 255)
            y = gt.mean()
            if y == 0:
                x = pred.mean()
                Q = 1.0 -x
            elif y == 1:
                x = pred.mean()
                Q = x
            else:
                gt[gt>=0.5] = 1
                gt[gt<0.5] = 0
                Q = 0.5 * S_object(pred, gt) + (1-0.5) * S_region(pred, gt)
                if Q.item() < 0:
                    Q = torch.FloatTensor([0.0])
            s_measure_q += Q.item()
            

            plt_image_skip = np.copy(plt_image)
            plt_image = np.ravel(plt_image)
            
            msk_skip = np.copy(msk)
            msk = np.ravel(msk)
            y_temp = (plt_image >= 0.5).astype(np.float32)
            tp = np.sum((y_temp * msk))
            # avoid prec becomes 0
            prec, recall = (tp + 1e-10) / (np.sum(y_temp) + 1e-10), (tp + 1e-10) / (np.sum(msk) + 1e-10)
            beta_square = 0.3
            f_score = (1 + beta_square) * prec * recall / (beta_square * prec + recall)
            
            mae = np.mean(np.abs(y_temp-msk))
            # if mae > 0.002:
            #     fig, ax = plt.subplots(1, 3, figsize=(15, 5))
            #     ax[0].imshow(np.squeeze(plt_image_skip), cmap='gray')
            #     ax[1].imshow(np.squeeze(msk_skip), cmap='gray')
            #     ax[2].imshow(img)
            #     ax[0].set_title(str(mae))
            #     plt.show()
            IoUs.append(f_score)
            maes.append(mae)
            
            
        # print(np.max(H), np.max(W))
        # assert(0)
        # print(all_stds)
        # assert(0)
        all_ious.append(np.mean(IoUs))
        all_maes.append(np.mean(maes))
  
    d[compact] = [all_ious, all_maes]
    ax[0].plot(segment_numbers, all_ious, label=f'{compact}')
    ax[0].scatter(segment_numbers, all_ious)
    for i, j in zip(segment_numbers, all_ious):
        ax[0].text(i, j+0.002, '{}'.format(i))

    ax[1].plot(segment_numbers, all_maes, label=f'{compact}')
    ax[1].scatter(segment_numbers, all_maes)
    for i, j in zip(segment_numbers, all_maes):
        ax[1].text(i, j+0.002, '{}'.format(i))

    print(all_ious, all_maes)
    print(torch.max(e_measure_scores)/num_images)
    print(s_measure_q/num_images)
    

    

# with open('segments_plot_data.pkl', 'wb') as f:
#     pickle.dump(d, f)
fs = 20
ax[0].set_title(f'Segmentation boundary intersection accuracy', fontsize=fs)
ax[0].set_xlabel('Segmentations', fontsize=fs)
ax[0].set_ylabel('Intersection Accuracy', fontsize=fs)
ax[0].set_xscale('log')
# ax[0].set_xticks(fontsize=fs, rotation=45)
# ax[0].set_yticks(fontsize=fs)
ax[0].legend(loc="lower right", fontsize=fs, title='Compactness', title_fontsize=fs)
for vertical in np.power(np.array(segment_numbers), 2):
    ax[0].axvline(x=vertical, linestyle='--')
# plt.savefig(f'compactness.jpg')
plt.show()
    

