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

class SPMNISTDataSet(torch.utils.data.Dataset):
    # images df, labels df, transforms
    # uses labels to determine if it needs to return X & y or just X in __getitem__
    def __init__(self, images, labels, num_seg, transforms=None):
        self.X = images
        self.y = labels
        self.transforms = transforms
        self.num_seg = num_seg
       
                    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, i):
        data = self.X.iloc[i, :] # gets the row
        # reshape the row into the image size 
        # (numpy arrays have the color channels dim last)
        data = np.array(data).astype(np.uint8).reshape(28, 28, 1)
        img_size = data.shape[1]
        # perform transforms if there are any
        if self.transforms:
            data = np.squeeze(self.transforms(data).numpy())

        segments = slic(data, n_segments=self.num_seg,
            compactness=COMPACTNESS,
            max_num_iter=10,
            convert2lab=False,
            enforce_connectivity=True,
            slic_zero=False,
            channel_axis=None
            )
        # if !test_set return the label as well, otherwise don't

        # plt.imshow(mark_boundaries(np.squeeze(data), segments, color=(1)), cmap='gray')
        # plt.show()

        regions = regionprops_table(segments, intensity_image=data, properties=('label', 'centroid', 'area',
                                                                                 'intensity_mean', 'coords'), extra_properties=[image_stdev])#, polarize])
                    
        features = np.zeros([self.num_seg, 5])
        label = regions['label']
        features[label-1, 0] = regions['centroid-0']
        features[label-1, 1] = regions['centroid-1']
        features[label-1, 2] = regions['area'] / (img_size**2)
        features[label-1, 3] = regions['intensity_mean']/255.
        features[label-1, 4] = regions['image_stdev']/255.


        features  = torch.tensor(features).float()

        return features, self.y.iloc[i]



class SPMNISTDataModule(pl.LightningDataModule):

    def __init__(self, **kwargs):
        super().__init__()

        train_transform_mnist = transforms.Compose(
                    [transforms.ToPILImage(),
                    transforms.Resize(32),
                    transforms.RandomAffine(degrees=20, translate=(0.1,0.1), scale=(0.9, 1.1)),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2),
                    transforms.ToTensor()
                    ])

        
        test_transform_mnist = transforms.Compose(
                        [transforms.ToPILImage(),
                        transforms.Resize(32),
                        transforms.ToTensor()
                        ])
        dataset_dir = kwargs.get('dataset_dir')
        self.batch_size = kwargs.get('batch_size')
        self.num_workers = kwargs.get('num_workers', 0)
        self.num_seg = kwargs.get('num_seg', 600)
        self.res = kwargs.get('size')

        train_set = pd.read_csv(f"{dataset_dir}/mnist_train.csv")
        test_set = pd.read_csv(f"{dataset_dir}/mnist_test.csv")

        train_images = train_set.iloc[:, 1:]
        train_labels = train_set.iloc[:, 0]
        
        test_images = test_set.iloc[:, 1:]
        test_labels = test_set.iloc[:, 0]

        mnist_train = SPMNISTDataSet(train_images, train_labels, self.num_seg,
        transforms=train_transform_mnist)

        self.train_source_loader = torch.utils.data.DataLoader(
            mnist_train,
            batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, drop_last=True
        )

        val_source_dataset = SPMNISTDataSet(test_images, test_labels, self.num_seg, transforms=test_transform_mnist)
        self.val_source_loader = torch.utils.data.DataLoader(
            val_source_dataset,
            batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, drop_last=False
        )


        
    def train_dataloader(self):
        return self.train_source_loader

    def val_dataloader(self):
        return self.val_source_loader

    def test_dataloader(self):
        return self.val_source_loader


