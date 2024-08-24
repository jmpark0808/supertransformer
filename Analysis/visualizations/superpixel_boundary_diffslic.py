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
from torchvision.transforms import ToTensor
import torch
from Blocks.diffslic_og import DiffSLIC, spixel_upsampling
import torch.nn.functional as F
from torch_geometric.utils import scatter

dataset_images = '/mnt/hdd/Datasets/DUTS/DUTS-TR/Image'
masks = '/mnt/hdd/Datasets/DUTS/DUTS-TR/Mask'
segment_numbers = [10, 15, 20, 25, 30, 40, 50, 100, 200, 300]
# segment_numbers = [100, 200, 300, 400, 500, 600, 800, 1000, 1500, 3000, 10000, 45000, 90000]
compactness = np.linspace(0.1, 2, 5)
d= {}
d['segment_numbers'] = segment_numbers
num_images = 3000

tt = ToTensor()

plt.figure(figsize=(10,10))

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
            img_np = np.array(img)
            
            img = tt(img)
            mask = tt(msk)
            msk = np.array(msk)
            
            
            # msk[msk>125] = 255
            # msk[msk<=125] = 0

            # empty_background = np.zeros_like(msk)

            # msk_boundaries = np.sum(mark_boundaries(empty_background, msk), axis=2)

            msk[msk<=125] = 0
            msk[msk>125] = 1
            
            num_seg = seg*seg
            

            with torch.no_grad():
                img = (img * 2 - 1).unsqueeze(0)
                
                h, w = img.shape[-2:]
                coords = torch.stack(torch.meshgrid(torch.linspace(-1, 1, h, dtype=torch.float),
                                                    torch.linspace(-1, 1, w, dtype=torch.float)), -1).unsqueeze(0)
                
                # sin embedding
                freqs = 2**torch.arange(2, dtype=torch.float)
                shape = coords.shape[:-1] + (-1,)
                scaled_x = (coords[..., None, :] * freqs[..., None]).reshape(shape) # (batch, *, n_points, num_feats * n_freq)
                scaled_x = torch.stack([scaled_x, scaled_x + 0.5 * torch.pi], -2).reshape(shape) # (batch, n_points, 2 * num_feats * n_freq)
               
                embedded_x = torch.sin(scaled_x).permute(0, 3, 1, 2) * compact
                
                # compute differentiable slic
                inputs = torch.cat([img, embedded_x], 1).cuda()
                diff_slic = DiffSLIC(num_seg, 5, 0.01, 1, stable=True)
      
                feats, assign, p2s = diff_slic(inputs)
                
            
                
              
                assert assign.shape[-2:] == inputs.shape[-2:], f"{assign.shape[-2:]} v.s., {inputs.shape[-2:]}"
                # assignment to label
                h_s, w_s = feats.shape[-2:]
                
                hard_assign = F.one_hot(assign.argmax(1), (2 * 1 + 1)**2).permute(0, 3, 1, 2).contiguous().float()
                
                label = torch.arange(h_s * w_s, dtype=torch.float).reshape(1, 1, h_s, w_s).cuda()
                label = spixel_upsampling(label, hard_assign, candidate_radius=1).long()

                # seg (bs, 1, H, W)
                # img (bs, 3, H, W)
                # mask (bs, 1, H, W)

                _, h, w = mask.size()
                b = 1
       

                shift = torch.arange(0, b, device=label.device).repeat_interleave(h*w)*num_seg
                
                seg_ = label.reshape(-1)+shift
                mask = mask.reshape(-1)
    
                
  
                seq_mask = scatter(mask.cuda(), seg_, reduce='mean', dim_size=num_seg*b)
                # colour = seg.matmul(img) # BS*H*W x 3
                # centroid = seg.matmul(coord) # BS*H*W x 2
                
                
                
                seq_mask = seq_mask.reshape(b, num_seg)
                
            segments = label.detach().cpu().numpy().squeeze()

            
            # regions = regionprops_table(segments, intensity_image=img_np, properties=('label', 'centroid'))#, polarize])
            # centers_y = regions['centroid-0']
            # centers_x = regions['centroid-1']
            # plt.imshow(mark_boundaries(img_np, segments))
            # plt.scatter(centers_x, centers_y, c='blue', s=30)
            # for ind, (x, y) in enumerate(zip(centers_x, centers_y)):
            #     plt.text(x, y, str(regions['label'][ind]))
            # plt.show()

            
            plt_image = seq_mask.detach().cpu().numpy().reshape(-1)[segments.reshape(-1)].reshape([img.shape[2], img.shape[3]])
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

    
with open('segments_plot_data_diffslic.pkl', 'wb') as f:
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
    

