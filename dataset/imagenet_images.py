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
import matplotlib.pyplot as plt
from scipy import sparse as sp
from scipy.spatial.distance import pdist, squareform
import pandas as pd
from skimage.segmentation import mark_boundaries
import torchvision 
import xml.etree.ElementTree as ET
import pathlib
import scipy
import pickle


        


class ImageNetDatasetTest(data.Dataset):
    def __init__(self, root_dir, transforms, class_to_idx):
        self.root_dir = root_dir
        self.image_list = sorted(os.listdir('{}/Data/CLS-LOC/val'.format(root_dir)))
        self.target_list = sorted(os.listdir('{}/Annotations/CLS-LOC/val'.format(root_dir)))
        self.transform = transforms
        self.class_to_idx = class_to_idx
  



    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, item):
        img_name = '{}/Data/CLS-LOC/val/{}'.format(self.root_dir, self.image_list[item])
        target_name = '{}/Annotations/CLS-LOC/val/{}'.format(self.root_dir, self.target_list[item]) 

        target = ET.parse(target_name)
        root = target.getroot()
        target = root[5][0].text
        target = self.class_to_idx[target]

        
        img = Image.open(img_name)
        img = img.convert('RGB')
        img = self.transform(img)

        features, target = torch.tensor(img).float(), torch.tensor(target)

        return features, target



class ImageNetDatasetTrain(torchvision.datasets.ImageFolder):
    def __init__(self, root, transform) -> None:
        super().__init__(root, transform=transform)


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


        if self.target_transform is not None:
            target = self.target_transform(target)


        features, target = torch.tensor(img).float(), torch.tensor(target)

        return features, target



        

class ImageNetDataModule(pl.LightningDataModule):

    def __init__(self, **kwargs):
        super().__init__()

        
        train_dir = kwargs.get('train_dir')
        test_dir = kwargs.get('test_dir')
        self.batch_size = kwargs.get('batch_size')
        self.num_workers = kwargs.get('num_workers', 0)
        self.seed = kwargs.get('seed')
        self.size = kwargs.get('size')
        generator = torch.Generator().manual_seed(self.seed)

        transform_train = transforms.Compose([transforms.RandomAffine(10, (0.1, 0.1), (0.9, 1.1)),
                                         transforms.Resize((self.size, self.size)),
                                           transforms.ToTensor()])
        transform_val_test = transforms.Compose([transforms.Resize((self.size, self.size)),
                                           transforms.ToTensor()])

        train_dataset = ImageNetDatasetTrain(train_dir, transform_train)
        class_to_idx = train_dataset.class_to_idx
        train_size = int(0.8*len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size], generator=generator)
        val_dataset.dataset.transform = transform_val_test

        test_dataset = ImageNetDatasetTest(test_dir, transform_val_test, class_to_idx)

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


if __name__ == "__main__":
    transform_train = transforms.Compose([transforms.RandomAffine(10, (0.1, 0.1), (0.9, 1.1)),
                                         transforms.Resize((320, 320)),
                                           transforms.ToTensor()])
    transform_val_test = transforms.Compose([transforms.Resize((320, 320)),
                                        transforms.ToTensor()])

    train_dir = '/mnt/dragon/Datasets/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/train'
    train_dataset = ImageNetDatasetTrain(train_dir, transform_train)
    class_to_idx = train_dataset.class_to_idx
    train_size = int(0.8*len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    val_dataset.dataset.transform = transform_val_test
    test_dir = '/mnt/dragon/Datasets/imagenet-object-localization-challenge/ILSVRC/'
    test_dataset = ImageNetDatasetTest(test_dir, transform_val_test, class_to_idx)

    # for batch in val_dataset:
    #     imgs, label = batch
    #     plt.imshow(imgs.permute(1, 2, 0).detach().cpu().numpy())
    #     plt.show()

