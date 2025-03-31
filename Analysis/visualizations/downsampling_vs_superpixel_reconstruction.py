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
import matplotlib as mpl
load_data = True
if not load_data:


    dataset_images = '/mnt/hdd/Datasets/DUTS/DUTS-TR/Image'
    masks = '/mnt/hdd/Datasets/DUTS/DUTS-TR/Mask'
    # segment_numbers = [224, 448]# , 300
    # segment_numbers = [100, 200, 300, 400, 500, 600, 800, 1000, 1500, 3000, 10000, 45000, 90000]
    
    compactness = [0.1, 1, 10, 50]
    
    num_images = 3000
    ds_maes = []
    for file in tqdm(os.listdir(dataset_images)[:num_images]):

        name = file.split('.jpg')[0]
        image = os.path.join(dataset_images, name+'.jpg')


        img = Image.open(image).resize((224, 224))
        img_array = np.array(img)

        img_downsample = img.resize((56, 56))
        img_upsample = img_downsample.resize((img_array.shape[1], img_array.shape[0]))


        img_upsample_array = np.array(img_upsample)

        downsample_mae = np.mean(np.abs(img_array-img_upsample_array))


        ds_maes.append(downsample_mae)
    
    all_sp_maes = []
    for compact in compactness:
        
        seg = 3136
        sp_maes = []
        

        for file in tqdm(os.listdir(dataset_images)[:num_images]):

            name = file.split('.jpg')[0]
            image = os.path.join(dataset_images, name+'.jpg')


            img = Image.open(image).resize((224, 224))
            img_array = np.array(img)



            # fig = plt.figure(figsize=(15, 10))
            # spec = mpl.gridspec.GridSpec(ncols=3, nrows=4, wspace=0.1,hspace=0.3,left=0.01,right=0.99,top=0.95,bottom=0.01)

            # ax1 = fig.add_subplot(spec[1:3,0])
            # ax2 = fig.add_subplot(spec[0:2,1])
            # ax3 = fig.add_subplot(spec[2:4,1])
            # ax4 = fig.add_subplot(spec[0:2,2])
            # ax5 = fig.add_subplot(spec[2:4,2])
       
            # ax1.imshow(np.array(img))
            # ax1.set_title('Original', fontsize=15)
            # ax2.imshow(img_upsample)
            # ax2.set_title('Downsample Boundaries', fontsize=15)
            # ax2.hlines(list(range(img_array.shape[1]//32, img_array.shape[1], img_array.shape[1]//32)), xmin=0, xmax=img_array.shape[1]-1,colors='y')
            # ax2.vlines(list(range(img_array.shape[0]//32, img_array.shape[0], img_array.shape[0]//32)), ymin=0, ymax=img_array.shape[0]-1, colors='y')
            # ax4.imshow(img_upsample)
            # ax4.set_title('Downsample Reconstruction', fontsize=15)
           

            segments = slic(img_array, n_segments=seg,
            compactness=compact,
            max_num_iter=10,
            convert2lab=True,
            enforce_connectivity=False,
            slic_zero=False)


            regions = regionprops_table(segments, img_array, properties=('label', 'centroid', 'area', 'intensity_mean',
                                                                                'coords',))
            

            seq_mask = np.zeros([max(regions['label']), 3])
 

            for ind, r, g, b in zip(regions['label'], regions['intensity_mean-0'], regions['intensity_mean-1'], regions['intensity_mean-2']):
                seq_mask[ind-1, 0] = r
                seq_mask[ind-1, 1] = g
                seq_mask[ind-1, 2] = b


            reconstruct_sp_image = seq_mask[segments-1, :].reshape([img_array.shape[0], img_array.shape[1], 3])
            
            # ax3.set_title('Superpixel Boundary', fontsize=15)
            # ax3.imshow(mark_boundaries(reconstruct_sp_image/255., segments))
            # ax5.imshow(reconstruct_sp_image/255.)
            # ax5.set_title('Superpixel Reconstruction', fontsize=15)

            # ax1.axis('off')
            # ax2.axis('off')
            # ax3.axis('off')
            # ax4.axis('off')
            # ax5.axis('off')

            # fig.savefig(f'./{name}sp.pdf', format='pdf')
            # plt.show()
            sp_mae = np.mean(np.abs(img_array-reconstruct_sp_image))
            

            sp_maes.append(sp_mae)

        all_sp_maes.append(sp_maes)


    
    np.save('./Analysis/reconstruction_error_sp.npy', np.array(all_sp_maes))
    np.save('./Analysis/reconstruction_error_ds.npy', np.array(ds_maes))
else:
    sp_maes = np.load('./Analysis/reconstruction_error_sp.npy')
    ds_maes = np.load('./Analysis/reconstruction_error_ds.npy')
    
    
    bins =np.histogram(np.hstack((sp_maes[0], sp_maes[1], sp_maes[2], sp_maes[3],ds_maes)), bins=100)[1]

    counts2, _ = np.histogram(sp_maes[0], bins=bins)
    counts3, _ = np.histogram(sp_maes[1], bins=bins)
    counts4, _ = np.histogram(sp_maes[2], bins=bins)
    counts5, _ = np.histogram(sp_maes[3], bins=bins)
    counts6, _ = np.histogram(ds_maes, bins=bins)

    print(np.mean(sp_maes), np.mean(ds_maes))
    plt.figure(figsize=(12,5))
    labels= ['Superpixel C 0.1', 'Superpixel C 1', 'Superpixel C 10', 'Superpixel C 50', 'Downsample']
    colors = ['blue', 'orange', 'green', 'red', 'purple']
    bplot = plt.boxplot([sp_maes[0], sp_maes[1], sp_maes[2], sp_maes[3], ds_maes], patch_artist=True, tick_labels=labels, showfliers=False)
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
    # plt.hist(sp_maes[0], label='Superpixel C 0.1', bins=bins, alpha=0.5)
    # plt.hist(sp_maes[1], label='Superpixel C 1', bins=bins, alpha=0.5)
    # plt.hist(sp_maes[2], label='Superpixel C 10', bins=bins, alpha=0.5)
    # plt.hist(sp_maes[3], label='Superpixel C 50', bins=bins, alpha=0.5)
    # plt.hist(ds_maes, label='Downsample', bins=bins)
    # plt.xlabel('Mean Absolute Error', fontsize=20)
    plt.yscale('log')
    plt.xticks(fontsize=15)
    plt.ylabel('Mean Absolute Error', fontsize=15)
    plt.title('Reconstruction Error', fontsize=15)
    # plt.legend(fontsize=20)
    plt.savefig(f'/mnt/hdd/Figures/SuperFormer/hist_sp.pdf', format='pdf')
    plt.show()
    
  
   
