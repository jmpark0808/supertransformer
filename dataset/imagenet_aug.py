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
import time
import pickle

class ImageNetDataset(torchvision.datasets.ImageFolder):
    def __init__(self, root, transform,  num_seg, compactness, coeff, ignore_phase) -> None:
        super().__init__(root, transform=transform)
        self.tensor = transforms.ToTensor()
        self.num_seg = num_seg
        self.coeff = coeff
        self.compactness = compactness
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
        


    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """

        img, target = self.imgs[index], self.targets[index]
        
        
        img = Image.open(img[0])
        img = img.convert('RGB')
        

        img = self.transform(img)

        img_np = np.array(img).transpose(1, 2, 0)
        
    
        img_size = img_np.shape


        segments = slic(img_np, n_segments=self.num_seg,
            compactness=self.compactness,
            max_num_iter=10,
            convert2lab=True,
            enforce_connectivity=False,
            slic_zero=False)

                
        regions = regionprops_table(segments, intensity_image=img_np, properties=('label', 'centroid', 'intensity_mean'), extra_properties=[image_stdev, self.fourier_descriptors])#, polarize])

        seq_len = len(regions['label'])
        label = regions['label']
        features = np.zeros([self.num_seg, 8+(self.coeff)*2])
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

        if self.target_transform is not None:
            target = self.target_transform(target)


        target = torch.tensor(target)

        return features, target

        

class SPImageNetAugDataModule(pl.LightningDataModule):

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
        self.ignore_phase = kwargs.get('ignore_phase')
        self.size = kwargs.get('size')


        train_transform = transforms.Compose(
                    [transforms.Resize([self.size, self.size]),
                     transforms.RandomAffine(degrees=20, translate=(0.1,0.1), scale=(0.9, 1.1)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor()
                    ])
        FFT_transform = []
        # val_test_transform = None
        
        val_test_transform = transforms.Compose(
                        [transforms.Resize([self.size, self.size]),
                         transforms.ToTensor()
                        ])
        generator = torch.Generator().manual_seed(self.seed)

        train_dataset = ImageNetDataset(train_dir, train_transform,  self.num_seg, self.compactness, self.coeff,
                                          self.ignore_phase)
        class_to_idx = train_dataset.class_to_idx
        train_size = int(0.8*len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size], generator=generator)
        val_dataset.dataset.transform = val_test_transform
       

        test_dataset = ImageNetDataset(test_dir, val_test_transform, self.num_seg,  self.compactness, self.coeff,
                                          self.ignore_phase)

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


