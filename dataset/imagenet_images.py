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



from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, DEFAULT_CROP_PCT
from timm.data.auto_augment import rand_augment_transform, augment_and_mix_transform, auto_augment_transform
from timm.data.transforms import RandomResizedCropAndInterpolation, ToNumpy, ToTensor
from timm.data.random_erasing import RandomErasing

def create_transform(
        input_size,
        use_prefetcher=False,
        scale=None,
        ratio=None,
        hflip=0.5,
        vflip=0.,
        color_jitter=0.4,
        auto_augment=None,
        interpolation='bilinear',
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
        re_prob=0.,
        re_mode='const',
        re_count=1,
        re_num_splits=0,
        separate=False):

    if isinstance(input_size, (tuple, list)):
        img_size = input_size[-2:]
    else:
        img_size = input_size

    
    transform = transforms_imagenet_train(
        img_size,
        scale=scale,
        ratio=ratio,
        hflip=hflip,
        vflip=vflip,
        color_jitter=color_jitter,
        auto_augment=auto_augment,
        interpolation=interpolation,
        use_prefetcher=use_prefetcher,
        mean=mean,
        std=std,
        re_prob=re_prob,
        re_mode=re_mode,
        re_count=re_count,
        re_num_splits=re_num_splits,
        separate=separate)

    return transform

def transforms_imagenet_train(
        img_size=224,
        scale=None,
        ratio=None,
        hflip=0.5,
        vflip=0.,
        color_jitter=0.4,
        auto_augment=None,
        interpolation='random',
        use_prefetcher=False,
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
        re_prob=0.,
        re_mode='const',
        re_count=1,
        re_num_splits=0,
        separate=False,
):
    """
    If separate==True, the transforms are returned as a tuple of 3 separate transforms
    for use in a mixing dataset that passes
     * all data through the first (primary) transform, called the 'clean' data
     * a portion of the data through the secondary transform
     * normalizes and converts the branches above with the third, final transform
    """
    scale = tuple(scale or (0.08, 1.0))  # default imagenet scale range
    ratio = tuple(ratio or (3./4., 4./3.))  # default imagenet ratio range
    primary_tfl = [
        RandomResizedCropAndInterpolation(img_size, scale=scale, ratio=ratio, interpolation=interpolation)]
    if hflip > 0.:
        primary_tfl += [transforms.RandomHorizontalFlip(p=hflip)]
    if vflip > 0.:
        primary_tfl += [transforms.RandomVerticalFlip(p=vflip)]

    secondary_tfl = []
    if auto_augment:
        assert isinstance(auto_augment, str)
        if isinstance(img_size, (tuple, list)):
            img_size_min = min(img_size)
        else:
            img_size_min = img_size
        aa_params = dict(
            translate_const=int(img_size_min * 0.45),
            img_mean=tuple([min(255, round(255 * x)) for x in mean]),
        )
        
        if auto_augment.startswith('rand'):
            secondary_tfl += [rand_augment_transform(auto_augment, aa_params)]
        elif auto_augment.startswith('augmix'):
            aa_params['translate_pct'] = 0.3
            secondary_tfl += [augment_and_mix_transform(auto_augment, aa_params)]
        else:
            secondary_tfl += [auto_augment_transform(auto_augment, aa_params)]
    elif color_jitter is not None:
        # color jitter is enabled when not using AA
        if isinstance(color_jitter, (list, tuple)):
            # color jitter should be a 3-tuple/list if spec brightness/contrast/saturation
            # or 4 if also augmenting hue
            assert len(color_jitter) in (3, 4)
        else:
            # if it's a scalar, duplicate for brightness, contrast, and saturation, no hue
            color_jitter = (float(color_jitter),) * 3
        secondary_tfl += [transforms.ColorJitter(*color_jitter)]

    final_tfl = []
    if use_prefetcher:
        # prefetcher and collate will handle tensor conversion and norm
        final_tfl += [ToNumpy()]
    else:
        final_tfl += [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=torch.tensor(mean),
                std=torch.tensor(std))
        ]
        if re_prob > 0.:
            final_tfl.append(
                RandomErasing(re_prob, mode=re_mode, max_count=re_count, num_splits=re_num_splits, device='cpu'))

    if separate:
        return transforms.Compose(primary_tfl), transforms.Compose(secondary_tfl), transforms.Compose(final_tfl)
    else:
        return transforms.Compose(primary_tfl + secondary_tfl + final_tfl)
        


# class ImageNetDatasetTest(data.Dataset):
#     def __init__(self, root_dir, transforms, class_to_idx):
#         self.root_dir = root_dir
#         self.image_list = sorted(os.listdir('{}/Data/CLS-LOC/val'.format(root_dir)))
#         self.target_list = sorted(os.listdir('{}/Annotations/CLS-LOC/val'.format(root_dir)))
#         self.transform = transforms
#         self.class_to_idx = class_to_idx
  



#     def __len__(self):
#         return len(self.image_list)

#     def __getitem__(self, item):
#         img_name = '{}/Data/CLS-LOC/val/{}'.format(self.root_dir, self.image_list[item])
#         target_name = '{}/Annotations/CLS-LOC/val/{}'.format(self.root_dir, self.target_list[item]) 

#         target = ET.parse(target_name)
#         root = target.getroot()
#         target = root[5][0].text
#         target = self.class_to_idx[target]

        
#         img = Image.open(img_name)
#         img = img.convert('RGB')
#         img = self.transform(img)

#         features, target = torch.tensor(img).float(), torch.tensor(target)

#         return features, target



class ImageNetDataset(torchvision.datasets.ImageFolder):
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


        target = torch.tensor(target)

        return img, target



        

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

        resize_im = self.size
        
            # this should always dispatch to transforms_imagenet_train
        transform_train = create_transform(
            input_size=self.size,
            color_jitter=0.4,
            auto_augment='rand-m9-mstd0.5-inc1',
            re_prob=0.25,
            re_mode='pixel',
            re_count=1,
            interpolation='bicubic',
        )
            
        # transform_train = transforms.Compose([transforms.RandomAffine(10, (0.1, 0.1), (0.9, 1.1)),
        #                                  transforms.Resize((self.size, self.size)),
        #                                    transforms.ToTensor()])
        
        t = []
        
        size = int((256 / 224) * self.size)
        t.append(
            transforms.Resize(size),
            # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(self.size))
           

        t.append(transforms.ToTensor())
        t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
        transform_val_test =  transforms.Compose(t)

        #  = transforms.Compose([transforms.Resize((self.size, self.size)),
        #                                    transforms.ToTensor()])

        train_dataset = ImageNetDataset(train_dir, transform_train)
        class_to_idx = train_dataset.class_to_idx
        train_size = int(0.8*len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size], generator=generator)
        val_dataset.dataset.transform = transform_val_test

        test_dataset = ImageNetDataset(test_dir, transform_val_test)

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

