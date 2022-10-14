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

def image_stdev(region, intensities):
    # note the ddof arg to get the sample var if you so desire!
    return np.std(intensities[region])

def polarize(region):
    # note the ddof arg to get the sample var if you so desire!
    centroid = np.mean(np.nonzero(region),axis=1)
    coords = np.nonzero(region)
    normalized_coords = np.stack([coords[0]-centroid[0], coords[1]-centroid[1]], axis=1)
    rho = np.linalg.norm(normalized_coords, axis=1)
    phi = np.arctan2(normalized_coords[:, 0], normalized_coords[:, 1])*180/np.pi+180
    radii_max = np.zeros([NUM_CHUNK, 2])
    radii_min = np.zeros([NUM_CHUNK, 2])

    chunk = CHUNK
    
    for ind, degree in enumerate(range(0, 360, chunk)):
        try:
            radii_max[ind] = normalized_coords[np.argmax(np.where((degree<=phi) & (phi<degree+chunk), rho, np.zeros_like(rho)))]
        except: 
            pass
        
        try:
            radii_min[ind] = normalized_coords[np.argmin(np.where((degree<=phi) & (phi<degree+chunk), rho, np.inf*np.ones_like(rho)))]
        except: 
            pass
        
        
    return np.concatenate((radii_max, radii_min), axis=0)


def embed(region, intensities):
    # note the ddof arg to get the sample var if you so desire!
    # cut_out = np.zeros([24, 24])
    # h,w = region.shape
    # indices = np.nonzero(region)
    # if h <=24 and h<= 24: # Fits inside square 
    #     start_h = (24-h)//2
    #     start_w = (24-w)//2
    #     cut_out[indices[0]+start_h,indices[1]+start_w] = intensities[indices]
    # else:
        

    # indices = np.nonzero(region)

    cut_out = np.zeros([49, 49])
    cut_out[np.nonzero(region)] = intensities[np.nonzero(region)]
    return (cut_out.reshape(-1))
    

def lbp(region, intensities):
    (hist, _) = np.histogram(intensities[region].ravel(),
			bins=np.arange(0, 57 + 3),
			range=(0, 57 + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    return hist


class ToTensorSP(object):
    def __init__(self, num_seg):
        self.tensor = transforms.ToTensor()
        self.num_seg = num_seg

    def __call__(self, sample):
        img, mask = sample['image'], sample['mask']
        img_np = np.array(img)
        img_size = img_np.shape[1]
        mask_np = np.array(mask)/255.
        segments = slic(img_np, n_segments=self.num_seg,
            compactness=COMPACTNESS,
            max_num_iter=10,
            convert2lab=True,
            enforce_connectivity=False,
            slic_zero=True, min_size_factor=0.)
        # slic = SlicAvx2(num_components=self.num_seg, compactness=10)
        # segments = slic.iterate(img_np)

        # vs_right = np.vstack([segments[:,:-1].ravel(), segments[:,1:].ravel()])
        # vs_below = np.vstack([segments[:-1,:].ravel(), segments[1:,:].ravel()])
        # vs_diagonal_r = np.vstack([segments[:-1,:-1].ravel(), segments[1:,1:].ravel()])
        # vs_diagonal_l = np.vstack([segments[1:,:-1].ravel(), segments[:-1,1:].ravel()])
        # bneighbors = np.unique(np.hstack([vs_right, vs_below, vs_diagonal_r, vs_diagonal_l]), axis=1)
    

        regions = regionprops_table(segments, intensity_image=img_np, properties=('label', 'centroid', 'area', 'intensity_mean',
                                                                                    'extent', 'coords', 'eccentricity'), extra_properties=[image_stdev])#, polarize])
                    
        seq_len = len(regions['label'])
        features = np.zeros([self.num_seg, 11])
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
        features[label-1, 8] = regions['image_stdev-0']/255.
        features[label-1, 9] = regions['image_stdev-1']/255.
        features[label-1, 10] = regions['image_stdev-2']/255.
        # for i in range(NUM_CHUNK*2):
        #     features[label-1, 11+i] = regions[f'polarize-{i}-0']
        #     features[label-1, 11+NUM_CHUNK*2+i] = regions[f'polarize-{i}-1']

        for ind, coord in zip(regions['label'], regions['coords']):
            seq_mask[ind-1] = np.sum(mask_np[coord[:, 0], coord[:, 1]])/len(coord[:, 0])

        # neighbor_array = np.zeros([self.num_seg, self.num_seg])
        # neighbor_array[bneighbors[0]-1, bneighbors[1]-1] = 1
        # neighbor_array[bneighbors[1]-1, bneighbors[0]-1] = 1
        distances = euclidean_distances(features[:, :2], features[:, :2])
        ind = np.argsort(distances, axis=1)
        neighbor_array = ind <= NUM_NEIGHBOURS

        features, neighbor_array, seq_mask, segments, mask, img = torch.tensor(features).float(), torch.tensor(neighbor_array).float(), torch.tensor(seq_mask).float(), torch.tensor(segments), self.tensor(mask), self.tensor(img)
        return {'features': features, 'seq_mask': seq_mask, 'segments': segments, 'mask': mask, 'img': img, 'neighbor_array': neighbor_array}

class ToTensorSPET(object):
    def __init__(self, num_seg):
        self.tensor = transforms.ToTensor()
        self.num_seg = num_seg

    def __call__(self, sample):
        img, mask = sample['image'], sample['mask']
        img_np = np.array(img)
        img_size = img_np.shape[1]
        mask_np = np.array(mask)/255.
        segments = slic(img_np, n_segments=self.num_seg,
            compactness=COMPACTNESS,
            max_num_iter=10,
            convert2lab=True,
            enforce_connectivity=False,
            slic_zero=False, min_size_factor=0.)
        # slic = SlicAvx2(num_components=self.num_seg, compactness=10)
        # segments = slic.iterate(img_np)

        # vs_right = np.vstack([segments[:,:-1].ravel(), segments[:,1:].ravel()])
        # vs_below = np.vstack([segments[:-1,:].ravel(), segments[1:,:].ravel()])
        # vs_diagonal_r = np.vstack([segments[:-1,:-1].ravel(), segments[1:,1:].ravel()])
        # vs_diagonal_l = np.vstack([segments[1:,:-1].ravel(), segments[:-1,1:].ravel()])
        # bneighbors = np.unique(np.hstack([vs_right, vs_below, vs_diagonal_r, vs_diagonal_l]), axis=1)
    

        regions = regionprops_table(segments, intensity_image=img_np, properties=('label', 'centroid', 'area', 'intensity_mean',
                                                                                    'extent', 'coords', 'eccentricity'), extra_properties=[image_stdev, polarize])
                    
        seq_len = len(regions['label'])
        features = np.zeros([self.num_seg, 11+NUM_CHUNK*4])
        seq_mask = np.zeros([self.num_seg])
        label = regions['label']
        features[label-1, 0] = regions['centroid-0']
        features[label-1, 1] = regions['centroid-1']
        features[label-1, 2] = regions['area'] / (img_size**2)
        features[label-1, 3] = regions['intensity_mean-0']/255.
        features[label-1, 4] = regions['intensity_mean-1']/255.
        features[label-1, 5] = regions['intensity_mean-2']/255.
        features[label-1, 6] = regions['extent']
        features[label-1, 7] = regions['eccentricity']
        features[label-1, 8] = regions['image_stdev-0']/255.
        features[label-1, 9] = regions['image_stdev-1']/255.
        features[label-1, 10] = regions['image_stdev-2']/255.
        for i in range(NUM_CHUNK*2):
            features[label-1, 11+i] = regions[f'polarize-{i}-0']
            features[label-1, 11+NUM_CHUNK*2+i] = regions[f'polarize-{i}-1']

        for ind, coord in zip(regions['label'], regions['coords']):
            seq_mask[ind-1] = np.sum(mask_np[coord[:, 0], coord[:, 1]])/len(coord[:, 0])

        distances = euclidean_distances(features[:, :2], features[:, :2])
        # ind = np.argsort(distances, axis=1)
        # ranged_ind = np.array(range(self.num_seg))
        # ranged_ind = np.tile(ranged_ind, (self.num_seg, 1)) 
        # ind = np.take_along_axis(ranged_ind, ind, axis=1)
        # ind = ind[:, :NUM_NEIGHBOURS]
        ind = distances <= 30
  
        features, neighbor_array, seq_mask, segments, mask, img = torch.tensor(features).float(), torch.tensor(ind).long(), torch.tensor(seq_mask).float(), torch.tensor(segments), self.tensor(mask), self.tensor(img)
        return {'features': features, 'seq_mask': seq_mask, 'segments': segments, 'mask': mask, 'img': img, 'neighbor_array': neighbor_array}

class ToTensorSPEmbed(object):
    def __init__(self, num_seg):
        self.tensor = transforms.ToTensor()
        self.num_seg = num_seg

    def __call__(self, sample):
        img, mask = sample['image'], sample['mask']
        img_np = np.array(img)
        img_size = img_np.shape[1]
        mask_np = np.array(mask)/255.
        segments = slic(img_np, n_segments=self.num_seg,
            compactness=COMPACTNESS,
            max_num_iter=10,
            convert2lab=True,
            enforce_connectivity=False,
            slic_zero=False, min_size_factor=0.)
    

        regions = regionprops_table(segments, intensity_image=img_np, properties=('label', 'centroid','coords'), extra_properties=[embed])
                    
        features = np.zeros([self.num_seg, 2+2401*3])
        seq_mask = np.zeros([self.num_seg])
        label = regions['label']
        features[label-1, 0] = regions['centroid-0']
        features[label-1, 1] = regions['centroid-1']
        for i in range(3):
            for j in range(2401):
                features[label-1, i*2401+j] = regions[f'embed-{j}-{i}']

        for ind, coord in zip(regions['label'], regions['coords']):
            seq_mask[ind-1] = np.sum(mask_np[coord[:, 0], coord[:, 1]])/len(coord[:, 0])

        distances = euclidean_distances(features[:, :2], features[:, :2])
        # ind = np.argsort(distances, axis=1)
        # ranged_ind = np.array(range(self.num_seg))
        # ranged_ind = np.tile(ranged_ind, (self.num_seg, 1)) 
        # ind = np.take_along_axis(ranged_ind, ind, axis=1)
        # ind = ind[:, :NUM_NEIGHBOURS]
        ind = distances <= 20
  
        features, neighbor_array, seq_mask, segments, mask, img = torch.tensor(features).float(), torch.tensor(ind).long(), torch.tensor(seq_mask).float(), torch.tensor(segments), self.tensor(mask), self.tensor(img)
        return {'features': features, 'seq_mask': seq_mask, 'segments': segments, 'mask': mask, 'img': img, 'neighbor_array': neighbor_array}

class ToTensorSPCNN(object):
    def __init__(self, num_seg):
        self.tensor = transforms.ToTensor()
        self.num_seg = num_seg

    def __call__(self, sample):
        img, mask = sample['image'], sample['mask']


        img_lab = np.array(color.rgb2lab(img))
        img_hsv = np.array(img.convert('HSV'))
        img_gray = np.array(img.convert('L'))
        img_np = np.array(img.convert('RGB'))
        img_size = img_np.shape[1]
        mask_np = np.array(mask)/255.
        segments = slic(img_np, n_segments=self.num_seg,
            compactness=10,
            max_num_iter=10,
            convert2lab=True,
            enforce_connectivity=False,
            slic_zero=True, min_size_factor=0.)
        # slic = SlicAvx2(num_components=self.num_seg, compactness=10)
        # segments = slic.iterate(img_np)
    
        lbp_np = local_binary_pattern(img_gray, 57, 8)
        regions = regionprops_table(segments, intensity_image=img_np, properties=('label', 'centroid', 'intensity_mean','coords'))

        regions_lbp = regionprops_table(segments, intensity_image=lbp_np, extra_properties=[lbp])
        regions_lab = regionprops_table(segments, intensity_image=img_lab, properties=('label', 'intensity_mean'))
        regions_hsv = regionprops_table(segments, intensity_image=img_hsv, properties=('label', 'intensity_mean'))
 
        seq_len = len(regions['label'])
        features = np.zeros([self.num_seg, 70])
        seq_mask = np.zeros([self.num_seg])
        label = regions['label']
        features[label-1, 0] = regions['centroid-0']/300.
        features[label-1, 1] = regions['centroid-1']/300.
        features[label-1, 2] = regions['intensity_mean-0']/255.
        features[label-1, 3] = regions['intensity_mean-1']/255.
        features[label-1, 4] = regions['intensity_mean-2']/255.
        features[label-1, 5] = regions_lab['intensity_mean-0']
        features[label-1, 6] = regions_lab['intensity_mean-1']
        features[label-1, 7] = regions_lab['intensity_mean-2']
        features[label-1, 8] = regions_hsv['intensity_mean-0']
        features[label-1, 9] = regions_hsv['intensity_mean-1']
        features[label-1, 10] = regions_hsv['intensity_mean-2']

        for ind in range(59):
            features[label-1, ind+11] = regions_lbp[f'lbp-{ind}']


        for ind, coord in zip(regions['label'], regions['coords']):
            seq_mask[ind-1] = np.sum(mask_np[coord[:, 0], coord[:, 1]])/len(coord[:, 0])



        features, seq_mask, segments, mask, img = torch.tensor(features).float(), torch.tensor(seq_mask).float(), torch.tensor(segments), self.tensor(mask), self.tensor(img)
        return {'features': features, 'seq_mask': seq_mask, 'segments': segments, 'mask': mask, 'img': img}

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


class SPEDataset(data.Dataset):
    def __init__(self, root_dir, num_seg, data_augmentation=True):
        self.root_dir = root_dir
        self.image_list = sorted(os.listdir('{}/Image'.format(root_dir)))
        self.mask_list = sorted(os.listdir('{}/Mask'.format(root_dir)))
        self.transform = transforms.Compose(
            [RandomFlip(0.5),
             RandomCrop(300, 350),
             ToTensorSPET(num_seg)])
        if not data_augmentation:
            self.transform = transforms.Compose([Resize(300), ToTensorSPET(num_seg)])

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


class SPEmbedDataset(data.Dataset):
    def __init__(self, root_dir, num_seg, data_augmentation=True):
        self.root_dir = root_dir
        self.image_list = sorted(os.listdir('{}/Image'.format(root_dir)))
        self.mask_list = sorted(os.listdir('{}/Mask'.format(root_dir)))
        self.transform = transforms.Compose(
            [RandomFlip(0.5),
             RandomCrop(300, 350),
             ToTensorSPEmbed(num_seg)])
        if not data_augmentation:
            self.transform = transforms.Compose([Resize(300), ToTensorSPEmbed(num_seg)])

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


class SPCNNDataset(data.Dataset):
    def __init__(self, root_dir, num_seg, data_augmentation=True):
        self.root_dir = root_dir
        self.image_list = sorted(os.listdir('{}/Image'.format(root_dir)))
        self.mask_list = sorted(os.listdir('{}/Mask'.format(root_dir)))
        self.transform = transforms.Compose(
            [RandomFlip(0.5),
             RandomCrop(300, 350),
             ToTensorSPCNN(num_seg)])
        if not data_augmentation:
            self.transform = transforms.Compose([Resize(300), ToTensorSPCNN(num_seg)])

        self.root_dir = root_dir
        self.data_augmentation = data_augmentation

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, item):
        img_name = '{}/Image/{}'.format(self.root_dir, self.image_list[item])
        mask_name = '{}/Mask/{}'.format(self.root_dir, self.mask_list[item])
        img = Image.open(img_name)
        mask = Image.open(mask_name)
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


class SPEDataModule(pl.LightningDataModule):

    def __init__(self, **kwargs):
        super().__init__()

        self.train_dir = kwargs.get('dataset_tr')
        self.val_dir = kwargs.get('dataset_val')
        self.test_dir = kwargs.get('dataset_test')
        self.batch_size = kwargs.get('batch_size')
        self.num_workers = kwargs.get('num_workers', 0)
        self.num_seg = kwargs.get('num_seg', 600)

        
    def train_dataloader(self):
        data_train = SPEDataset(self.train_dir, self.num_seg, True)
        return DataLoader(
                data_train, batch_size=self.batch_size, 
                num_workers=self.num_workers, shuffle=True, pin_memory=False)

    def val_dataloader(self):
        data_val = SPEDataset(self.val_dir, self.num_seg, False)
        return DataLoader(
                data_val, batch_size=self.batch_size, 
                num_workers=self.num_workers, pin_memory=False)

    def test_dataloader(self):
        data_test = SPEDataset(self.test_dir, self.num_seg, False)
        return DataLoader(
                data_test, batch_size=self.batch_size, 
                num_workers=self.num_workers, pin_memory=False)



class SPEmbedDataModule(pl.LightningDataModule):

    def __init__(self, **kwargs):
        super().__init__()

        self.train_dir = kwargs.get('dataset_tr')
        self.val_dir = kwargs.get('dataset_val')
        self.test_dir = kwargs.get('dataset_test')
        self.batch_size = kwargs.get('batch_size')
        self.num_workers = kwargs.get('num_workers', 0)
        self.num_seg = kwargs.get('num_seg', 600)

        
    def train_dataloader(self):
        data_train = SPEmbedDataset(self.train_dir, self.num_seg, True)
        return DataLoader(
                data_train, batch_size=self.batch_size, 
                num_workers=self.num_workers, shuffle=True, pin_memory=False)

    def val_dataloader(self):
        data_val = SPEmbedDataset(self.val_dir, self.num_seg, False)
        return DataLoader(
                data_val, batch_size=self.batch_size, 
                num_workers=self.num_workers, pin_memory=False)

    def test_dataloader(self):
        data_test = SPEmbedDataset(self.test_dir, self.num_seg, False)
        return DataLoader(
                data_test, batch_size=self.batch_size, 
                num_workers=self.num_workers, pin_memory=False)


class SPCNNDataModule(pl.LightningDataModule):

    def __init__(self, **kwargs):
        super().__init__()

        self.train_dir = kwargs.get('dataset_tr')
        self.val_dir = kwargs.get('dataset_val')
        self.test_dir = kwargs.get('dataset_test')
        self.batch_size = kwargs.get('batch_size')
        self.num_workers = kwargs.get('num_workers', 0)
        self.num_seg = kwargs.get('num_seg', 600)

        
    def train_dataloader(self):
        data_train = SPCNNDataset(self.train_dir, self.num_seg, True)
        return DataLoader(
                data_train, batch_size=self.batch_size, 
                num_workers=self.num_workers, shuffle=True, pin_memory=False)

    def val_dataloader(self):
        data_val = SPCNNDataset(self.val_dir, self.num_seg, False)
        return DataLoader(
                data_val, batch_size=self.batch_size, 
                num_workers=self.num_workers, pin_memory=False)

    def test_dataloader(self):
        data_test = SPCNNDataset(self.test_dir, self.num_seg, False)
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