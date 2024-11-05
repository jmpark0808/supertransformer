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
import time
from fast_slic.avx2 import SlicAvx2
import torch
from torch_geometric.utils import scatter
import torch.nn.functional as F
from torch_scatter import scatter_std

dataset_images = '/mnt/hdd/Datasets/DUTS/DUTS-TR/Image'
masks = '/mnt/hdd/Datasets/DUTS/DUTS-TR/Mask'
# segment_numbers = [224, 448]# , 300
# segment_numbers = [100, 200, 300, 400, 500, 600, 800, 1000, 1500, 3000, 10000, 45000, 90000]
segment_numbers = [12544]
compactness = [10]
d= {}
d['segment_numbers'] = segment_numbers
num_images = 3000
use_pickle = False


from dataset.superpixel import SPDataset
from torch.utils.data import DataLoader

import os
train_dir = '/home/eddie/Datasets/EORSSD/TR'

batch_size = 1
num_workers = 20


dataloader = 'SP'

coeff = 10
ignore_phase = False

image_list = np.array(sorted([os.path.join('{}/Image'.format(train_dir), f) for f in os.listdir('{}/Image'.format(train_dir))]))[:num_images]
mask_list = np.array(sorted([os.path.join('{}/Mask'.format(train_dir), f) for f in os.listdir('{}/Mask'.format(train_dir))]))[:num_images]


indices = np.array(list(range(len(image_list))))
np.random.shuffle(indices)

tr_image_list = image_list
tr_mask_list = mask_list











fig, ax = plt.subplots(1, 2, figsize=(10, 10))

for compact in tqdm(compactness):
    all_ious = []
    all_maes = []
    for seg in segment_numbers:
        IoUs = []
        maes = []
        dataset = SPDataset(tr_image_list, tr_mask_list, seg, 448, compact, False, dataloader, coeff, ignore_phase)
        dl = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
        # for file in tqdm(os.listdir(dataset_images)[:num_images]):
        all_stds = 0
        count = 0
        for batch in tqdm(dl):
            # name = file.split('.jpg')[0]
            # image = os.path.join(dataset_images, name+'.jpg')
            # mask = os.path.join(masks, name+'.png')

            # img = Image.open(image)
            # msk = Image.open(mask)
            # img = img.convert('RGB').resize((300, 300))
            # msk = msk.convert('L').resize((300, 300))
            # img = np.array(img)
            # msk = np.array(msk)
            
            # msk[msk>125] = 255
            # msk[msk<=125] = 0

            # empty_background = np.zeros_like(msk)

            # msk_boundaries = np.sum(mark_boundaries(empty_background, msk), axis=2)

            # msk[msk<=125] = 0
            # msk[msk>125] = 1
            
            num_seg = seg*seg
            # start = time.time()
            # segments = slic(img, n_segments=num_seg,
            # compactness=compact,
            # max_num_iter=10,
            # convert2lab=True,
            # enforce_connectivity=False,
            # slic_zero=False)
            # slic = SlicAvx2(num_components=num_seg, compactness=compact, min_size_factor=0.)
            # segments = slic.iterate(img)

            # end = time.time()
            # ms = end-start
            # print(ms)
            

            # segments = slic(image=img, n_segments=seg, compactness=compact, min_size_factor=0.5, max_num_iter=3, enforce_connectivity=False)
            # segments = slic.iterate(img)

            # superpixel_boundaries = np.sum(mark_boundaries(empty_background, segments), axis=2)

            # iou = np.sum(np.logical_and((msk_boundaries == 2),(superpixel_boundaries == 2)))/np.sum(msk_boundaries>0)
            # regions = regionprops_table(segments, img, properties=('label', 'centroid', 'area', 'intensity_mean',
            #                                                                      'coords',))
            # end = time.time()
            # ms = end-start
            
            # try:
            #     max(regions['label'])
            # except:
            #     plt.imshow(img)
            #     plt.show()
            # seq_mask = np.zeros([max(regions['label'])])
            # # assert len(regions['label']) == max(regions['label']), 'Wrong number of labels'

            # for ind, coord in zip(regions['label'], regions['coords']):
            #     seq_mask[ind-1] = np.sum(msk[coord[:, 0], coord[:, 1]])/len(coord[:, 0])



            # plt_image = seq_mask[segments-1].reshape([img.shape[0], img.shape[1]])
            # plt_image = np.ravel(plt_image)

            img = batch['features'].cuda()
            segments = batch['segments']
            msk = batch['mask'].detach().numpy()

            # ax[0].imshow(batch['features'].squeeze().permute(1, 2, 0).detach().cpu().numpy())
            # ax[1].imshow(mark_boundaries(batch['features'].squeeze().permute(1, 2, 0).detach().cpu().numpy(), segments.squeeze().detach().cpu().numpy()))
            # plt.show()
            
            with torch.no_grad():

                
                seg_ = segments.cuda().reshape(-1).long()
                mask = torch.tensor(msk, device='cuda')
                mask = F.interpolate(mask, (segments.size(1), segments.size(2)))
                mask = mask.reshape(-1)
                red = img[0, 0, :, :].reshape(-1)
                green = img[0, 1, :, :].reshape(-1)
                blue = img[0, 2, :, :].reshape(-1)
    
                

                seq_mask = scatter(mask, seg_, reduce='mean')
                red_std = scatter_std(red, seg_)
                green_std = scatter_std(green, seg_)
                blue_std = scatter_std(blue, seg_)

            all_stds += torch.mean(red_std) + torch.mean(green_std) + torch.mean(blue_std)
            count +=1
            
            
            
            # seq_mask = seq_mask.reshape(1, segments.size(1), segments.size(2))
            # seq_mask = 
            seq_mask = seq_mask.reshape(-1)
            
            
    
        
            plt_image = seq_mask.reshape(-1)[segments.reshape(-1)].reshape([1, 1, segments.shape[1], segments.shape[2]])
            plt_image = F.interpolate(plt_image, (msk.shape[2], msk.shape[3])).detach().cpu().numpy()
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
            # if f_score < 0.9:
            #     fig, ax = plt.subplots(1, 2)
            #     ax[0].imshow(np.squeeze(plt_image_skip), cmap='gray')
            #     ax[1].imshow(np.squeeze(msk_skip), cmap='gray')
            #     plt.show()
            mae = np.mean(np.abs(plt_image-msk))
            
            IoUs.append(f_score)
            maes.append(mae)

        all_stds /= count
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

    

    

with open('segments_plot_data.pkl', 'wb') as f:
    pickle.dump(d, f)
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
    

