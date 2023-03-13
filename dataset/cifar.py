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


class CIFARDataset(torchvision.datasets.CIFAR10):
    def __init__(self, root, num_seg, train, transform, target_transform, download) -> None:
        super().__init__(root, train, transform, target_transform, download)
        self.num_seg = num_seg
    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        
        img = np.transpose(img.cpu().numpy(), (1, 2, 0))
        img_size = img.shape[1]

        segments = slic(img, n_segments=self.num_seg,
            compactness=COMPACTNESS,
            max_num_iter=10,
            convert2lab=False,
            enforce_connectivity=True,
            slic_zero=False,
            )
        # if !test_set return the label as well, otherwise don't

        # plt.imshow(mark_boundaries(img, segments))
        # plt.show()

        regions = regionprops_table(segments, intensity_image=img, properties=('label', 'centroid', 'area',
                                                                                 'intensity_mean', 'coords'), extra_properties=[image_stdev])#, polarize])
                    
        features = np.zeros([self.num_seg, 9])
        label = regions['label']
        features[label-1, 0] = regions['centroid-0']
        features[label-1, 1] = regions['centroid-1']
        features[label-1, 2] = regions['area'] / (img_size**2)
        features[label-1, 3] = regions['intensity_mean-0']/255.
        features[label-1, 4] = regions['intensity_mean-1']/255.
        features[label-1, 5] = regions['intensity_mean-2']/255.
        features[label-1, 6] = regions['image_stdev-0']/255.
        features[label-1, 7] = regions['image_stdev-1']/255.
        features[label-1, 8] = regions['image_stdev-2']/255.


        features  = torch.tensor(features).float()

        return features, target
        

class SPCIFARDataModule(pl.LightningDataModule):

    def __init__(self, **kwargs):
        super().__init__()

        train_transform = transforms.Compose(
                    [transforms.RandomAffine(degrees=20, translate=(0.1,0.1), scale=(0.9, 1.1)),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2),
                    transforms.ToTensor()
                    ])

        
        test_transform = transforms.Compose(
                        [transforms.ToTensor()
                        ])
        dataset_dir = kwargs.get('dataset_dir')
        self.batch_size = kwargs.get('batch_size')
        self.num_workers = kwargs.get('num_workers', 0)
        self.num_seg = kwargs.get('num_seg', 600)
        self.res = kwargs.get('size')


        self.train_source_loader = torch.utils.data.DataLoader(
            CIFARDataset(dataset_dir, self.num_seg, True, train_transform, None, True),
            batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, drop_last=True
        )

        self.val_source_loader = torch.utils.data.DataLoader(
            CIFARDataset(dataset_dir, self.num_seg, False, test_transform, None, True),
            batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, drop_last=False
        )


        
    def train_dataloader(self):
        return self.train_source_loader

    def val_dataloader(self):
        return self.val_source_loader

    def test_dataloader(self):
        return self.val_source_loader


