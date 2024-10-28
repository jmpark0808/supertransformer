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



dataset_images = '/mnt/f/Datasets/DUTS/DUTS-TR/Image'
masks = '/mnt/f/Datasets/DUTS/DUTS-TR/Mask'
# segment_numbers = [224, 448]# , 300
# segment_numbers = [100, 200, 300, 400, 500, 600, 800, 1000, 1500, 3000, 10000, 45000, 90000]
segment_numbers = [1024]
compactness = [10]
d= {}
d['segment_numbers'] = segment_numbers
num_images = 3000
use_pickle = False

 

for compact in tqdm(compactness):
    all_sp_maes = []
    all_ds_maes = []
    for seg in segment_numbers:
        sp_maes = []
        ds_maes = []

        for file in tqdm(os.listdir(dataset_images)[:num_images]):

            name = file.split('.jpg')[0]
            image = os.path.join(dataset_images, name+'.jpg')
   

            img = Image.open(image)
            img_array = np.array(img)

            img_downsample = img.resize((32, 32))
            img_upsample = img_downsample.resize((img_array.shape[1], img_array.shape[0]))
            plt.imshow(img_upsample)
            plt.show()

            
            img_upsample_array = np.array(img_upsample)

            downsample_mae = np.mean(np.abs(img_array-img_upsample_array))

            

            segments = slic(img_array, n_segments=seg,
            compactness=compact,
            max_num_iter=10,
            convert2lab=True,
            enforce_connectivity=False,
            slic_zero=False)


            regions = regionprops_table(segments, img_array, properties=('label', 'centroid', 'area', 'intensity_mean',
                                                                                 'coords',))
            
            try:
                max(regions['label'])
            except:
                plt.imshow(img)
                plt.show()
            seq_mask = np.zeros([max(regions['label']), 3])
            # assert len(regions['label']) == max(regions['label']), 'Wrong number of labels'

            for ind, r, g, b in zip(regions['label'], regions['intensity_mean-0'], regions['intensity_mean-1'], regions['intensity_mean-2']):
                seq_mask[ind-1, 0] = r
                seq_mask[ind-1, 1] = g
                seq_mask[ind-1, 2] = b



            reconstruct_sp_image = seq_mask[segments-1, :].reshape([img_array.shape[0], img_array.shape[1], 3])
            plt.imshow(reconstruct_sp_image/255.)
            plt.show()
            sp_mae = np.mean(np.abs(img_array-reconstruct_sp_image))
            
            # assert(0)
            sp_maes.append(sp_mae)
            ds_maes.append(downsample_mae)

        print(np.mean(sp_maes), np.mean(ds_maes))    
        all_sp_maes.append(np.mean(sp_maes))
        all_ds_maes.append(np.mean(ds_maes))
  
   
