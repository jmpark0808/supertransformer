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

dataset_images = '/mnt/hdd/Datasets/DUTS/DUTS-TR/Image'
masks = '/mnt/hdd/Datasets/DUTS/DUTS-TR/Mask'
segment_numbers = [32, 56, 112]# , 300
# segment_numbers = [100, 200, 300, 400, 500, 600, 800, 1000, 1500, 3000, 10000, 45000, 90000]
compactness = [10]
d= {}
d['segment_numbers'] = segment_numbers
num_images = 3000
use_pickle = False


from dataset.superpixel import SPDataset
from torch.utils.data import DataLoader

import os
train_dir = '/mnt/dragon/Datasets/EORSSD/TE/'

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











fig, ax = plt.subplots(1, 2, figsize=(10, 20))

for compact in tqdm(compactness):
    all_ious = []
    all_maes = []
    for seg in segment_numbers:
        IoUs = []
        maes = []
        dataset = SPDataset(tr_image_list, tr_mask_list, seg*seg, 224, compact, False, dataloader, coeff, ignore_phase)
        dl = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
        # for file in tqdm(os.listdir(dataset_images)[:num_images]):
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

            segments = batch['segments']
            msk = batch['mask'].detach().numpy()

            # ax[0].imshow(batch['features'].squeeze().permute(1, 2, 0).detach().cpu().numpy())
            # ax[1].imshow(mark_boundaries(batch['features'].squeeze().permute(1, 2, 0).detach().cpu().numpy(), segments.squeeze().detach().cpu().numpy()))
            # plt.show()

            with torch.no_grad():

                
                seg_ = segments.cuda().reshape(-1).long()
                mask = torch.tensor(msk, device='cuda').reshape(-1)
    
                

                seq_mask = scatter(mask, seg_, reduce='mean')
            
            
            
            
            seq_mask = seq_mask.reshape(-1)
            
    
        
            plt_image = seq_mask.detach().cpu().numpy().reshape(-1)[segments.reshape(-1)].reshape([msk.shape[2], msk.shape[3]])
            plt_image = np.ravel(plt_image)
                

            msk = np.ravel(msk)
            y_temp = (plt_image >= 0.5).astype(np.float)
            tp = np.sum((y_temp * msk))
            # avoid prec becomes 0
            prec, recall = (tp + 1e-10) / (np.sum(y_temp) + 1e-10), (tp + 1e-10) / (np.sum(msk) + 1e-10)
            beta_square = 0.3
            f_score = (1 + beta_square) * prec * recall / (beta_square * prec + recall)

            mae = np.mean(np.abs(plt_image-msk))
            
            IoUs.append(f_score)
            maes.append(mae)

        all_ious.append(np.mean(IoUs))
        all_maes.append(np.mean(maes))
  
    d[compact] = [all_ious, all_maes]
    ax[0].plot(np.power(np.array(segment_numbers), 2), all_ious, label=f'{compact}')
    ax[0].scatter(np.power(np.array(segment_numbers), 2), all_ious)
    for i, j in zip(np.power(np.array(segment_numbers), 2), all_ious):
        ax[0].text(i, j+0.002, '{}'.format(i))

    ax[1].plot(np.power(np.array(segment_numbers), 2), all_maes, label=f'{compact}')
    ax[1].scatter(np.power(np.array(segment_numbers), 2), all_maes)
    for i, j in zip(np.power(np.array(segment_numbers), 2), all_maes):
        ax[1].text(i, j+0.002, '{}'.format(i))

    

    

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
# plt.savefig(f'compactness.jpg')
plt.show()
    

