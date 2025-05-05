import os
import torch.utils.data as data
import torchvision.transforms as transforms
from collections import defaultdict
import numpy as np
import torch
from PIL import Image, ImageCms
from skimage.segmentation import slic
from skimage.measure import regionprops_table
from skimage.feature import local_binary_pattern
from sklearn.metrics.pairwise import euclidean_distances
from skimage import color
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from fast_slic.avx2 import SlicAvx2
from dataset.constants import *
import matplotlib.pyplot as plt
from scipy import sparse as sp
from scipy.spatial.distance import pdist, squareform
from dataset.attributes import *
import pandas as pd
from skimage.segmentation import mark_boundaries
import torchvision 
import xml.etree.ElementTree as ET
from dataset.fft_transform import *
from dataset.moments_transform import *
from dataset.randaugment import RandAugment
import pathlib
import scipy
import pickle
from util.util import merge_contours, compute_central_moments


class ImageNetDataset(data.Dataset):
    def __init__(self, root_dir, augmentation, coeff, num_seg, size):
        self.root_dir = root_dir
        self.image_list = []
        self.target_list = []
        self.coeff = coeff
        self.size = size
        self.num_seg = num_seg
        self.augmentation = augmentation
        self.resample_points = int(((size**2)//num_seg)**0.5)*4

        ps = int((size*size//num_seg)**0.5)
        xs = np.arange(ps//2, ps//2+ps*56, ps)
        ys = np.arange(ps//2, ps//2+ps*56, ps)
        xv, yv = np.meshgrid(xs, ys, indexing='ij')
        self.grid = np.stack((xv, yv), axis=2)

      
        for file in os.listdir(root_dir):
            if '_target' in file:
                self.image_list.append(os.path.join(root_dir, file.split('_target')[0]+'.npy'))
                self.target_list.append(os.path.join(root_dir, file))  
            elif 'target' in file:
                self.image_list.append(os.path.join(root_dir, file.split('target')[0]+'.npy'))
                self.target_list.append(os.path.join(root_dir, file)) 
            else:
               continue
                

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, item):
        
        features = np.load(self.image_list[item])
        res = int(features.shape[0]**0.5)
        # Spatial augmentation
        features_amp = features[:, 8:8+(self.resample_points-1)]
        features_phase = features[:, 8+(self.resample_points-1):8+2*(self.resample_points-1)]
        moments = features[:, (8+2*(self.resample_points-1)):(16+2*(self.resample_points-1))]
        front = math.ceil(self.coeff/2.)
        back = self.coeff-front
        assert (front+back) <= (self.resample_points-1)
        colour = features[:, 2:8]
        centroid = features[:, 0:2] - self.grid.reshape(-1, 2)
        colour_and_centroid = np.concatenate((centroid, colour), axis=1)
        
        lbp = features[:, -10:]
        features_amp = np.concatenate((features_amp[:, :front], features_amp[:, -back:]), 1)
        features_phase = np.concatenate((features_phase[:, :front], features_phase[:, -back:]), 1)
        if self.augmentation:
            # moments = rotate_moments(moments, 0.5, 15)
            
            if self.coeff != 0:
                features_np = np.concatenate((colour_and_centroid, features_amp, features_phase, lbp), 1)
                features_np = horizontal_flip(features_np, self.coeff, 0.5, self.size, (int(self.num_seg**0.5), int(self.num_seg**0.5)))
            else:
                features_np = np.concatenate((colour_and_centroid, lbp), 1)
        else:
            if self.coeff != 0:
                features_np = np.concatenate((colour_and_centroid, features_amp, features_phase, lbp), 1)
            else:
                features_np = np.concatenate((colour_and_centroid, lbp), 1)
        #     centroids, colour, features_amp,features_phase, moments, lbp = horizontal_flip_moments(colour_and_centroid[:, :2], colour_and_centroid[:, 2:],
        #                                           features_amp, features_phase, moments, lbp, 0.5, self.size, (res, res))
        #     colour_and_centroid = np.concatenate((centroids, colour), 1)
        
        # if self.coeff == 0:
        #     moments_zeros = np.zeros_like(moments)
        #     features_np = np.concatenate((colour_and_centroid, moments_zeros, lbp), 1)
        # else:
        #     features_np = np.concatenate((colour_and_centroid, features_amp, features_phase, moments, lbp), 1)
            
        features = torch.tensor(features_np).float()
        # Colour augmentations
        if self.augmentation:
            randaug = RandAugment(5)
            erase = transforms.RandomErasing(0.25)
            color_space = features[:, 2:5].reshape(res, res, 3).permute(2, 0, 1)
            # fig, ax = plt.subplots(1, 2)
            # ax[0].imshow(color_space.permute(1, 2, 0).detach().cpu().numpy())
            # op_names = 'No augment'
            if np.random.random() < 0.5:
                color_space = (color_space*255).to(torch.uint8)
                color_space = randaug(color_space)
                color_space = color_space.float()
                color_space /= 255.
            color_space = erase(color_space)
            
            # ax[1].imshow(color_space.permute(1, 2, 0).detach().cpu().numpy())
            # ax[1].set_title(op_names)
            # plt.show()
            color_space = color_space.reshape(3, res*res).permute(1, 0)
            
            features[:, 2:5] = color_space


        target = torch.tensor(np.load(self.target_list[item]))

        return features, target
       


        


class ImageNetDatasetExport(torchvision.datasets.ImageFolder):
    def __init__(self, root, num_seg, coeff, size, compactness,
                  transform, export_dir, ignore_phase, enforce_connectivity) -> None:
        super().__init__(root, transform=transform)
        self.num_seg = num_seg
        self.compactness = compactness
        self.coeff = coeff
        self.export_dir = export_dir
        self.ignore_phase = ignore_phase
        self.enforce_connectivity = enforce_connectivity

        resample_points = int(((size**2)//num_seg)**0.5)*4
        self.resample_points = resample_points
        
        
        def fourier_descriptors(region):
            moments = compute_central_moments(region)
            region = (region*255).astype(np.uint8)
            contour, hierarchy = cv2.findContours(region, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            if len(contour)>1:
                merged_contour = merge_contours(contour)

                points = np.array(merged_contour).reshape((-1, 2)).astype(np.int32)

                indices_y = np.argwhere(points[:, 1]==np.min(points[:, 1])) # smallest y
                indices_x = np.argmin(points[indices_y, 0])
                points = np.roll(points, -indices_y[indices_x], axis=0)
            else:
                points = contour[0][:, 0, :]
            xi, yi = resample_2d(points, resample_points)
            contour_array = np.stack((xi, yi), axis=1)


            contour_complex = np.empty(contour_array.shape[:-1], dtype=complex)
            contour_complex.real = contour_array[:, 0]
            contour_complex.imag = contour_array[:, 1]
            fourier_result = np.fft.fft(contour_complex)[1:]


            amp = abs(fourier_result)
            phase = np.arctan2(fourier_result.imag, fourier_result.real)

            # return np.array(amp)
            return np.concatenate((amp, phase, moments))
        self.fourier_descriptors = fourier_descriptors

        def lbp(region, intensities):
            (hist, _) = np.histogram(intensities[region].ravel(),
                    bins=np.arange(0, 8+3),
                    range=(0, 8+2))
            hist = hist.astype("float")
            # hist /= (hist.sum() + 1e-7)
            return hist
        self.lbp = lbp


    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """

        img, target = self.imgs[index], self.targets[index]
        
        sp_file_name = pathlib.PureWindowsPath(rf'{img[0]}').as_posix().split('/')[-1].split('.')[0]+'.npy'
        sp_file_name_edge = pathlib.PureWindowsPath(rf'{img[0]}').as_posix().split('/')[-1].split('.')[0]+'edge.pickle'
        sp_file_target = pathlib.PureWindowsPath(rf'{img[0]}').as_posix().split('/')[-1].split('.')[0]+'_target.npy'
      

        sp_file_path = os.path.join(self.export_dir, sp_file_name)
        sp_file_path_edge_index = os.path.join(self.export_dir, sp_file_name_edge)
        sp_file_path_target = os.path.join(self.export_dir, sp_file_target)

        if os.path.exists(sp_file_path) and os.path.exists(sp_file_path_target):
            return torch.empty(0)
     

        
        img = Image.open(img[0])
        img_gray = img.convert('L')
        img = img.convert('RGB')
        

        if self.transform is not None:
            img = self.transform(img)
            img_gray = self.transform(img_gray)

        if self.target_transform is not None:
            target = self.target_transform(target)


        img_np = np.ascontiguousarray(np.transpose(img.cpu().numpy()*255, (1, 2, 0))).astype(np.uint8)
        img_gray_np = np.ascontiguousarray(np.squeeze(img_gray.cpu().numpy()*255)).astype(np.uint8)
        

        segments = slic(img_np, n_segments=self.num_seg,
            compactness=self.compactness,
            max_num_iter=10,
            convert2lab=True,
            enforce_connectivity=False,
            slic_zero=False)
        # slic = SlicAvx2(num_components=self.num_seg, compactness=self.compactness)
        # segments = slic.iterate(img_np)+1
        # plt.imshow(mark_boundaries(img_np, segments))
        # plt.show()
        # vs_right = np.vstack([segments[:,:-1].ravel(), segments[:,1:].ravel()])
        # vs_below = np.vstack([segments[:-1,:].ravel(), segments[1:,:].ravel()])
        # vs_diagonal_r = np.vstack([segments[:-1,:-1].ravel(), segments[1:,1:].ravel()])
        # vs_diagonal_l = np.vstack([segments[1:,:-1].ravel(), segments[:-1,1:].ravel()])
        # bneighbors = np.unique(np.hstack([vs_right, vs_below, vs_diagonal_r, vs_diagonal_l]), axis=1)
        # neighbor_array = np.zeros([self.num_seg, self.num_seg])
        # neighbor_array[bneighbors[0]-1, bneighbors[1]-1] = 1
        # neighbor_array[bneighbors[1]-1, bneighbors[0]-1] = 1
        # S = scipy.sparse.csr_matrix(neighbor_array)
        # file = open(sp_file_path_edge_index,'wb') #160kb
        # pickle.dump(S, file)
        # np.save(sp_file_path_edge_index, neighbor_array)
        lbp_np = local_binary_pattern(img_gray_np, 8, 1, method='uniform')
        regions_lbp = regionprops_table(segments, intensity_image=lbp_np, extra_properties=[self.lbp])
        regions = regionprops_table(segments, intensity_image=img_np, properties=('label', 'centroid', 'intensity_mean',
                                                                                    'coords'), extra_properties=[image_stdev, self.fourier_descriptors])#, polarize])


        seq_len = len(regions['label'])
        label = regions['label']
       
        
                
        features = np.zeros([self.num_seg, 8+(2*(self.resample_points-1))+8+10])
    
        
        for i in range((2*(self.resample_points-1))+8):
            features[label-1, 8+i] = regions[f'fourier_descriptors-{i}']

        
        features[label-1, 0] = regions['centroid-0']
        features[label-1, 1] = regions['centroid-1']
        features[label-1, 2] = regions['intensity_mean-0']/255.
        features[label-1, 3] = regions['intensity_mean-1']/255.
        features[label-1, 4] = regions['intensity_mean-2']/255.
        features[label-1, 5] = regions['image_stdev-0']/255.
        features[label-1, 6] = regions['image_stdev-1']/255.
        features[label-1, 7] = regions['image_stdev-2']/255.

        for ind in range(10):
            features[label-1, ind+8+(2*(self.resample_points-1))+8] = regions_lbp[f'lbp-{ind}']
    

        
        np.save(sp_file_path, features.astype(np.float16))
        np.save(sp_file_path_target, np.array([target]))

        return torch.empty(0)



        

class SPImageNetDataModule(pl.LightningDataModule):

    def __init__(self, **kwargs):
        super().__init__()

        
        train_dir = kwargs.get('train_dir')
        test_dir = kwargs.get('test_dir')
        self.batch_size = kwargs.get('batch_size')
        self.num_workers = kwargs.get('num_workers', 0)
        self.num_seg = kwargs.get('num_seg', 600)
        self.coeff = kwargs.get('coeff', 70)
        self.compactness = kwargs.get('compactness', 10)
        self.seed = kwargs.get('seed')
        self.dilation = kwargs.get('dilation')
        self.size = kwargs.get('size')

    

        train_dataset = ImageNetDataset(train_dir, True, self.coeff, self.num_seg, self.size)
        test_dataset = ImageNetDataset(test_dir, False, self.coeff, self.num_seg, self.size)

        self.train_source_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True,
                                                               num_workers =self.num_workers, drop_last=True)
        
        self.test_source_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False,
                                                              num_workers=self.num_workers, drop_last=False)

        
    def train_dataloader(self):
        return self.train_source_loader

    def val_dataloader(self):
        return self.test_source_loader

    def test_dataloader(self):
        return self.test_source_loader


