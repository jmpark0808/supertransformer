import os
import torch.utils.data as data
import torchvision.transforms as transforms
from collections import defaultdict
import numpy as np
import torch
from PIL import Image
from skimage.segmentation import slic
from skimage.measure import regionprops_table
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from fast_slic.avx2 import SlicAvx2

class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        img, mask = sample['image'], sample['mask']
        img, mask = img.resize((self.size, self.size), resample=Image.BILINEAR), mask.resize((self.size, self.size),
                                                                                             resample=Image.BILINEAR)
        return {'image': img, 'mask': mask}


class RandomCrop(object):
    def __init__(self, crop_size, resize_size):
        self.crop_size = crop_size
        self.resize_size = resize_size

    def __call__(self, sample):
        img, mask = sample['image'], sample['mask']
        img, mask = img.resize((self.resize_size, self.resize_size), resample=Image.BILINEAR), mask.resize((self.resize_size, self.resize_size), resample=Image.BILINEAR)
        h, w = img.size
        new_h, new_w = self.crop_size, self.crop_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)
        img = img.crop((left, top, left + new_w, top + new_h))
        mask = mask.crop((left, top, left + new_w, top + new_h))

        return {'image': img, 'mask': mask}


class RandomFlip(object):
    def __init__(self, prob):
        self.prob = prob
        self.flip = transforms.RandomHorizontalFlip(1.)

    def __call__(self, sample):
        if np.random.random_sample() < self.prob:
            img, mask = sample['image'], sample['mask']
            img = self.flip(img)
            mask = self.flip(mask)
            return {'image': img, 'mask': mask}
        else:
            return sample


class ToTensorSP(object):
    def __init__(self, num_seg):
        self.tensor = transforms.ToTensor()
        self.num_seg = num_seg

    def __call__(self, sample):
        img, mask = sample['image'], sample['mask']
        img_np = np.array(img)
        img_size = img_np.shape[1]
        mask_np = np.array(mask)/255.
        # segments = slic(img_np, n_segments=self.num_seg,
        #     compactness=10.0,
        #     max_num_iter=10,
        #     convert2lab=True,
        #     enforce_connectivity=False,
        #     slic_zero=True,
        #     min_size_factor=0.,)
        slic = SlicAvx2(num_components=self.num_seg, compactness=0.1, min_size_factor=0)
        segments = slic.iterate(img_np)

        vs_right = np.vstack([segments[:,:-1].ravel(), segments[:,1:].ravel()])
        vs_below = np.vstack([segments[:-1,:].ravel(), segments[1:,:].ravel()])
        bneighbors = np.unique(np.hstack([vs_right, vs_below]), axis=1)
 

        regions = regionprops_table(segments, intensity_image=img_np, properties=('label', 'centroid', 'area', 'intensity_mean', 'extent', 'coords', 'eccentricity'))
        seq_len = len(regions['label'])
        features = np.zeros([self.num_seg, 8])
        seq_mask = np.zeros([self.num_seg])
        label = regions['label']
        features[label-1, 0] = regions['centroid-0']/300.
        features[label-1, 1] = regions['centroid-1']/300.
        features[label-1, 2] = regions['area'] / (img_size**2)
        features[label-1, 3] = regions['intensity_mean-0']/255.
        features[label-1, 4] = regions['intensity_mean-1']/255.
        features[label-1, 5] = regions['intensity_mean-2']/255.
        features[label-1, 6] = regions['extent']
        features[label-1, 7] = regions['eccentricity']
        for ind, coord in zip(regions['label'], regions['coords']):
            seq_mask[ind-1] = np.sum(mask_np[coord[:, 0], coord[:, 1]])/len(coord[:, 0])

        neighbor_array = np.zeros([self.num_seg, self.num_seg])
        neighbor_array[bneighbors[0]-1, bneighbors[1]-1] = 1

        features, neighbor_array, seq_mask, segments, mask, img = torch.tensor(features).float(), torch.tensor(neighbor_array).float(), torch.tensor(seq_mask).float(), torch.tensor(segments), self.tensor(mask), self.tensor(img)
        return {'features': features, 'seq_mask': seq_mask, 'segments': segments, 'mask': mask, 'img': img, 'neighbor_array': neighbor_array}

class ToTensorRaw(object):
    def __init__(self):
        self.tensor = transforms.ToTensor()

    def __call__(self, sample):
        img, mask = sample['image'], sample['mask']
        img, mask = self.tensor(img), self.tensor(mask)
        return {'image': img, 'mask': mask}

class SPDataset(data.Dataset):
    def __init__(self, root_dir, num_seg, data_augmentation=True):
        self.root_dir = root_dir
        self.image_list = sorted(os.listdir('{}/Image'.format(root_dir)))
        self.mask_list = sorted(os.listdir('{}/Mask'.format(root_dir)))
        self.transform = transforms.Compose(
            [RandomFlip(0.5),
             RandomCrop(300, 350),
             ToTensorSP(num_seg)])
        if not data_augmentation:
            self.transform = transforms.Compose([Resize(300), ToTensorSP(num_seg)])

        self.root_dir = root_dir
        self.data_augmentation = data_augmentation

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, item):
        img_name = '{}/Image/{}'.format(self.root_dir, self.image_list[item])
        mask_name = '{}/Mask/{}'.format(self.root_dir, self.mask_list[item])
        img = Image.open(img_name)
        mask = Image.open(mask_name)
        img = img.convert('RGB')
        mask = mask.convert('L')
        sample = {'image': img, 'mask': mask}

        sample = self.transform(sample)
        return sample

class DUTSDataset(data.Dataset):
    def __init__(self, root_dir, size, train=True, data_augmentation=True):
        self.root_dir = root_dir
        self.image_list = sorted(os.listdir('{}/Image'.format(root_dir)))
        self.mask_list = sorted(os.listdir('{}/Mask'.format(root_dir)))
        self.transform = transforms.Compose(
            [RandomFlip(0.5),
             RandomCrop(size, int(size*1.2)),
             ToTensorRaw()])
        if not (train and data_augmentation):
            self.transform = transforms.Compose([Resize(size), ToTensorRaw()])
        self.root_dir = root_dir


    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, item):
        img_name = '{}/Image/{}'.format(self.root_dir, self.image_list[item])
        mask_name = '{}/Mask/{}'.format(self.root_dir, self.mask_list[item])
        img = Image.open(img_name)
        mask = Image.open(mask_name)
        img = img.convert('RGB')
        mask = mask.convert('L')
        sample = {'image': img, 'mask': mask}

        sample = self.transform(sample)
        return sample

class SPDataModule(pl.LightningDataModule):

    def __init__(self, **kwargs):
        super().__init__()

        self.train_dir = kwargs.get('dataset_tr')
        self.val_dir = kwargs.get('dataset_val')
        self.test_dir = kwargs.get('dataset_test')
        self.batch_size = kwargs.get('batch_size')
        self.num_workers = kwargs.get('num_workers', 0)
        self.num_seg = kwargs.get('num_seg', 600)

        
    def train_dataloader(self):
        data_train = SPDataset(self.train_dir, self.num_seg, True)
        return DataLoader(
                data_train, batch_size=self.batch_size, 
                num_workers=self.num_workers, shuffle=True, pin_memory=False)

    def val_dataloader(self):
        data_val = SPDataset(self.val_dir, self.num_seg, False)
        return DataLoader(
                data_val, batch_size=self.batch_size, 
                num_workers=self.num_workers, pin_memory=False)

    def test_dataloader(self):
        data_test = SPDataset(self.test_dir, self.num_seg, False)
        return DataLoader(
                data_test, batch_size=self.batch_size, 
                num_workers=self.num_workers, pin_memory=False)


class DUTSDataModule(pl.LightningDataModule):

    def __init__(self, **kwargs):
        super().__init__()

        self.train_dir = kwargs.get('dataset_tr')
        self.val_dir = kwargs.get('dataset_val')
        self.test_dir = kwargs.get('dataset_test')
        self.batch_size = kwargs.get('batch_size')
        self.num_workers = kwargs.get('num_workers', 0)
        self.image_size = kwargs.get('size')

        
    def train_dataloader(self):
        data_train = DUTSDataset(self.train_dir, self.image_size, True, True)
        return DataLoader(
                data_train, batch_size=self.batch_size, 
                num_workers=self.num_workers, shuffle=True, pin_memory=False)

    def val_dataloader(self):
        data_val = DUTSDataset(self.val_dir, self.image_size, False, False)
        return DataLoader(
                data_val, batch_size=self.batch_size, 
                num_workers=self.num_workers, pin_memory=False)

    def test_dataloader(self):
        data_test = DUTSDataset(self.test_dir, self.image_size, False, False)
        return DataLoader(
                data_test, batch_size=self.batch_size, 
                num_workers=self.num_workers, pin_memory=False)