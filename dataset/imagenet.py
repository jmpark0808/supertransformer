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
import pathlib
import scipy
import pickle
class ImageNetDataset(data.Dataset):
    def __init__(self, root_dir, augmentation, coeff, size):
        self.root_dir = root_dir
        self.image_list = []
        self.target_list = []
        self.coeff = coeff
        self.size = size
        self.augmentation = augmentation
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
        
        features_np = np.load(self.image_list[item])

        if self.augmentation:
            features_np = horizontal_flip(features_np, self.coeff, 0.5, self.size)

        features = torch.tensor(features_np).float()

        target = torch.tensor(np.load(self.target_list[item]))

        return features, target
       


        


class ImageNetDatasetTestExport(data.Dataset):
    def __init__(self, root_dir, transforms, num_seg, coeff, class_to_idx, compactness, export_dir, ignore_phase):
        self.root_dir = root_dir
        self.image_list = sorted(os.listdir('{}/Data/CLS-LOC/val'.format(root_dir)))
        self.target_list = sorted(os.listdir('{}/Annotations/CLS-LOC/val'.format(root_dir)))
        self.transform = transforms
        self.class_to_idx = class_to_idx
        self.num_seg = num_seg
        self.compactness = compactness
        self.coeff = coeff
        self.export_dir = export_dir
        self.ignore_phase = ignore_phase

        def fourier_descriptors(region):
            region = (region*255).astype(np.uint8)
            contour, hierarchy = cv2.findContours(region, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            points = contour[0][:, 0, :]
            xi, yi = resample_2d(points, RESAMPLE_POINTS)
            contour_array = np.stack((xi, yi), axis=1)


            contour_complex = np.empty(contour_array.shape[:-1], dtype=complex)
            contour_complex.real = contour_array[:, 0]
            contour_complex.imag = contour_array[:, 1]
            fourier_result = np.fft.fft(contour_complex)

            fourier_result_front = fourier_result[1:1+coeff//2]
            fourier_result_back = fourier_result[-coeff//2:]
            fourier_result = np.concatenate((fourier_result_front, fourier_result_back), axis=0)

            amp = abs(fourier_result)
            phase = np.arctan2(fourier_result.imag, fourier_result.real)

            # return np.array(amp)
            return np.concatenate((amp, phase))

        self.fourier_descriptors = fourier_descriptors

        def lbp(region, intensities):
            (hist, _) = np.histogram(intensities[region].ravel(),
                    bins=np.arange(0, 8+3),
                    range=(0, 8+2))
            hist = hist.astype("float")
            # hist /= (hist.sum() + 1e-7)
            return hist
        self.lbp = lbp

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, item):
        img_name = '{}/Data/CLS-LOC/val/{}'.format(self.root_dir, self.image_list[item])
        target_name = '{}/Annotations/CLS-LOC/val/{}'.format(self.root_dir, self.target_list[item])

       

        sp_file_name = self.image_list[item].split('.')[0]+'.npy'
        sp_file_name_edge = self.image_list[item].split('.')[0]+'edge.pickle'
        sp_file_target = self.image_list[item].split('.')[0]+'_target.npy'
  
   
        sp_file_path = os.path.join(self.export_dir, sp_file_name)
        sp_file_path_edge_index = os.path.join(self.export_dir, sp_file_name_edge)
        sp_file_path_target = os.path.join(self.export_dir, sp_file_target)
    

        target = ET.parse(target_name)
        root = target.getroot()
        target = root[5][0].text
        target = self.class_to_idx[target]

        
        img = Image.open(img_name)
        img_gray = img.convert('L')
        img = img.convert('RGB')
        img = self.transform(img)
        img_gray = self.transform(img_gray)

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
        features = np.zeros([self.num_seg, 8+(self.coeff)*2+10])
    
        if self.ignore_phase:
            features = np.zeros([self.num_seg, 8+self.coeff])
            for i in range(self.coeff):
                features[label-1, 8+i] = regions[f'fourier_descriptors-{i}']
        else:
            for i in range(self.coeff*2):
                features[label-1, 8+i] = regions[f'fourier_descriptors-{i}']

        
        features[label-1, 0] = regions['centroid-0']
        features[label-1, 1] = regions['centroid-1']
        features[label-1, 2] = regions['intensity_mean-0']/255.
        features[label-1, 3] = regions['intensity_mean-1']/255.
        features[label-1, 4] = regions['intensity_mean-2']/255.
        features[label-1, 5] = regions['image_stdev-0']/255.
        features[label-1, 6] = regions['image_stdev-1']/255.
        features[label-1, 7] = regions['image_stdev-2']/255.

        for ind in range(8+2):
            features[label-1, ind+8+(self.coeff)*2] = regions_lbp[f'lbp-{ind}']
        

        np.save(sp_file_path, features)
        np.save(sp_file_path_target, np.array([target]))

        features, target = torch.tensor(features).float(), torch.tensor(target)

        return features, target



class ImageNetDatasetTrainExport(torchvision.datasets.ImageFolder):
    def __init__(self, root, num_seg, coeff, compactness, transform, mode, export_dir, ignore_phase) -> None:
        super().__init__(root, transform=transform)
        self.num_seg = num_seg
        self.compactness = compactness
        self.coeff = coeff
        self.mode = mode
        self.export_dir = export_dir
        self.ignore_phase = ignore_phase
        
        def fourier_descriptors(region):
            region = (region*255).astype(np.uint8)
            contour, hierarchy = cv2.findContours(region, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            points = contour[0][:, 0, :]
            xi, yi = resample_2d(points, RESAMPLE_POINTS)
            contour_array = np.stack((xi, yi), axis=1)


            contour_complex = np.empty(contour_array.shape[:-1], dtype=complex)
            contour_complex.real = contour_array[:, 0]
            contour_complex.imag = contour_array[:, 1]
            fourier_result = np.fft.fft(contour_complex)

            fourier_result_front = fourier_result[1:1+coeff//2]
            fourier_result_back = fourier_result[-coeff//2:]
            fourier_result = np.concatenate((fourier_result_front, fourier_result_back), axis=0)

            amp = abs(fourier_result)
            phase = np.arctan2(fourier_result.imag, fourier_result.real)

            # return np.array(amp)
            return np.concatenate((amp, phase))

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
        features = np.zeros([self.num_seg, 8+(self.coeff)*2+10])
    
        if self.ignore_phase:
            features = np.zeros([self.num_seg, 8+self.coeff])
            for i in range(self.coeff):
                features[label-1, 8+i] = regions[f'fourier_descriptors-{i}']
        else:
            for i in range(self.coeff*2):
                features[label-1, 8+i] = regions[f'fourier_descriptors-{i}']

        
        features[label-1, 0] = regions['centroid-0']
        features[label-1, 1] = regions['centroid-1']
        features[label-1, 2] = regions['intensity_mean-0']/255.
        features[label-1, 3] = regions['intensity_mean-1']/255.
        features[label-1, 4] = regions['intensity_mean-2']/255.
        features[label-1, 5] = regions['image_stdev-0']/255.
        features[label-1, 6] = regions['image_stdev-1']/255.
        features[label-1, 7] = regions['image_stdev-2']/255.

        for ind in range(8+2):
            features[label-1, ind+8+(self.coeff)*2] = regions_lbp[f'lbp-{ind}']
        

        
        np.save(sp_file_path, features)
        np.save(sp_file_path_target, np.array([target]))

        features, target = torch.tensor(features).float(), torch.tensor(target)

        return features, target



        

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
        generator = torch.Generator().manual_seed(self.seed)

        train_dataset = ImageNetDataset(train_dir, True, self.coeff, self.size)
        train_size = int(0.8*len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size], generator=generator)
        val_dataset.augmentation = False

        test_dataset = ImageNetDataset(test_dir, False, self.coeff, self.size)

        self.train_source_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True,
                                                               num_workers =self.num_workers, drop_last=True)
        
        self.val_source_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False,
                                                              num_workers=self.num_workers, drop_last=False)
        
        self.test_source_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False,
                                                              num_workers=self.num_workers, drop_last=False)

        
    def train_dataloader(self):
        return self.train_source_loader

    def val_dataloader(self):
        return [self.val_source_loader, self.test_source_loader]

    def test_dataloader(self):
        return self.test_source_loader


