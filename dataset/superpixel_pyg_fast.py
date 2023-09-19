import os
import torch.utils.data as data
import torchvision.transforms as transforms
from collections import defaultdict
import numpy as np
import torch
from PIL import Image, ImageCms
from skimage.segmentation import slic, mark_boundaries
from skimage.measure import regionprops_table
from skimage.feature import local_binary_pattern
from sklearn.metrics.pairwise import euclidean_distances
from torch_geometric.data import Data
from skimage import color
import pytorch_lightning as pl
from fast_slic.avx2 import SlicAvx2
from dataset.constants import *
import matplotlib.pyplot as plt
from scipy import sparse as sp
from scipy.spatial.distance import pdist, squareform
from dataset.attributes import *
from torch_geometric.loader import DataLoader
from pathlib import Path
import time

class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        img, mask = sample['image'], sample['mask']
        img, mask = img.resize((self.size, self.size), resample=Image.BILINEAR), mask.resize((self.size, self.size),
                                                                                             resample=Image.BILINEAR)
        return {'image': img, 'mask': mask}
    
class ResizeMask(object):
    def __init__(self, size):
        self.size = size
        self.tensor = transforms.ToTensor()

    def __call__(self, sample):
        mask = sample.resize((self.size, self.size), resample=Image.BILINEAR)
        return self.tensor(mask)


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

class RandomAffine(object):
    def __init__(self, rotate, translate, scale):
        self.rotate = rotate
        self.translate = translate
        self.scale = scale
                

    def __call__(self, sample):
        img, mask = sample['image'], sample['mask']
        H, W = img.size
        random_rotate = np.random.randint(-self.rotate, self.rotate)
        random_translate_x = int(np.random.random()*self.translate*W)
        random_translate_y = int(np.random.random()*self.translate*H)
        random_scale = 1+np.random.random()*2*self.scale-self.scale
        img = transforms.functional.affine(img, random_rotate, [random_translate_x, random_translate_y], random_scale, 0)
        mask = transforms.functional.affine(mask, random_rotate, [random_translate_x, random_translate_y], random_scale, 0)
        return {'image': img, 'mask': mask}

class RandomColorJitter(object):
    def __init__(self, brightness, contrast, saturation, hue) -> None:
        self.transform = transforms.ColorJitter(brightness, contrast, saturation, hue)

    def __call__(self, sample):
        img, mask = sample['image'], sample['mask']

        img = self.transform(img)

        return {'image': img, 'mask': mask}



class ToTensorSPFFT(object):
    def __init__(self, num_seg, compactness, coeff, ignore_phase, fully_connected):
        self.tensor = transforms.ToTensor()
        self.num_seg = num_seg
        self.coeff = coeff
        self.compactness = compactness
        self.ignore_phase = ignore_phase
        self.fully_connected = fully_connected
        
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


    def __call__(self, sample):
        img, mask = sample['image'], sample['mask']
        img_np = np.array(img)
        mask_np = np.array(mask)/255.
        img_size = img_np.shape

        # img_np = np.ascontiguousarray(np.transpose(img.cpu().numpy()*255, (1, 2, 0))).astype(np.uint8)
            
        slic = SlicAvx2(num_components=self.num_seg, compactness=self.compactness)
        segments = slic.iterate(img_np)
        # segments = slic(img_np, n_segments=self.num_seg,
        #     compactness=self.compactness,
        #     max_num_iter=3,
        #     convert2lab=True,
        #     enforce_connectivity=False,
        #     slic_zero=False)

        # plt.imshow(mark_boundaries(img_np, segments))
        # plt.show()

        # vs_right = np.vstack([segments[:,:-1].ravel(), segments[:,1:].ravel()])
        # vs_below = np.vstack([segments[:-1,:].ravel(), segments[1:,:].ravel()])
        # vs_diagonal_r = np.vstack([segments[:-1,:-1].ravel(), segments[1:,1:].ravel()])
        # vs_diagonal_l = np.vstack([segments[1:,:-1].ravel(), segments[:-1,1:].ravel()])
        # bneighbors = np.unique(np.hstack([vs_right, vs_below, vs_diagonal_r, vs_diagonal_l]), axis=1)
    

        regions = regionprops_table(segments, intensity_image=img_np, properties=('label', 'centroid', 'intensity_mean',
                                                                                    'coords'), extra_properties=[image_stdev, self.fourier_descriptors])#, polarize])

        seq_len = len(regions['label'])
        seq_mask = np.zeros([self.num_seg])
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
        


        for ind, coord in zip(regions['label'], regions['coords']):
            seq_mask[ind-1] = 1 if np.sum(mask_np[coord[:, 0], coord[:, 1]])/len(coord[:, 0]) >= 0.5 else 0


        # if self.fully_connected:
        #     neighbor_array = np.ones([self.num_seg, self.num_seg])
        # else:
        #     neighbor_array = np.zeros([self.num_seg, self.num_seg])
        #     neighbor_array[bneighbors[0]-1, bneighbors[1]-1] = 1
        #     neighbor_array[bneighbors[1]-1, bneighbors[0]-1] = 1
        # edge_index = np.nonzero(neighbor_array)



        # spatial_distances = euclidean_distances(features_centroids, features_centroids)
        # spatial_distances = spatial_distances[edge_index]
        
      
        # edge_features = np.expand_dims(spatial_distances, axis=1)
        

        # d = Data(x=torch.tensor(features).float(), edge_index=edge_index, edge_attr=edge_features)
        # seq_mask, segments, mask = torch.tensor(seq_mask).float(), torch.tensor(segments)

        return features, seq_mask, segments, self.tensor(mask)
    
class ToTensorSP(object):
    def __init__(self, num_seg, compactness, fully_connected):
        self.tensor = transforms.ToTensor()
        self.num_seg = num_seg
        self.compactness = compactness
        self.fully_connected =fully_connected
        


    def __call__(self, sample):
        img, mask = sample['image'], sample['mask']
        img_np = np.array(img)
        mask_np = np.array(mask)/255.
        img_size = img_np.shape

        # img_np = np.ascontiguousarray(np.transpose(img.cpu().numpy()*255, (1, 2, 0))).astype(np.uint8)
            
        slic = SlicAvx2(num_components=self.num_seg, compactness=self.compactness)
        segments = slic.iterate(img_np)
        # segments = slic(img_np, n_segments=self.num_seg,
        #     compactness=self.compactness,
        #     max_num_iter=3,
        #     convert2lab=True,
        #     enforce_connectivity=False,
        #     slic_zero=False)

        # plt.imshow(mark_boundaries(img_np, segments))
        # plt.show()
        # vs_right = np.vstack([segments[:,:-1].ravel(), segments[:,1:].ravel()])
        # vs_below = np.vstack([segments[:-1,:].ravel(), segments[1:,:].ravel()])
        # vs_diagonal_r = np.vstack([segments[:-1,:-1].ravel(), segments[1:,1:].ravel()])
        # vs_diagonal_l = np.vstack([segments[1:,:-1].ravel(), segments[:-1,1:].ravel()])
        # bneighbors = np.unique(np.hstack([vs_right, vs_below, vs_diagonal_r, vs_diagonal_l]), axis=1)

        regions = regionprops_table(segments, intensity_image=img_np, properties=('label', 'centroid', 'intensity_mean',
                                                                                    'coords'))#, polarize])

        seq_len = len(regions['label'])
        seq_mask = np.zeros([self.num_seg])
        label = regions['label']
        features = np.zeros([self.num_seg, 5])


    
        features[label-1, 0] = regions['centroid-0']
        features[label-1, 1] = regions['centroid-1']
        features[label-1, 2] = regions['intensity_mean-0']/255.
        features[label-1, 3] = regions['intensity_mean-1']/255.
        features[label-1, 4] = regions['intensity_mean-2']/255.      


        for ind, coord in zip(regions['label'], regions['coords']):
            seq_mask[ind-1] = 1 if np.sum(mask_np[coord[:, 0], coord[:, 1]])/len(coord[:, 0]) >= 0.5 else 0


        # if self.fully_connected:
        #     neighbor_array = np.ones([self.num_seg, self.num_seg])
        # else:
        #     neighbor_array = np.zeros([self.num_seg, self.num_seg])
        #     neighbor_array[bneighbors[0]-1, bneighbors[1]-1] = 1
        #     neighbor_array[bneighbors[1]-1, bneighbors[0]-1] = 1
        # edge_index = np.nonzero(neighbor_array)



        # spatial_distances = euclidean_distances(features_centroids, features_centroids)
        # spatial_distances = spatial_distances[edge_index]
        
      
        # edge_features = np.expand_dims(spatial_distances, axis=1)
        

        # d = Data(x=torch.tensor(features).float(), edge_index=edge_index, edge_attr=edge_features)
        # seq_mask, segments, mask = torch.tensor(seq_mask).float(), torch.tensor(segments)

        return features, seq_mask, segments, self.tensor(mask)



class SPDataset(data.Dataset):
    def __init__(self, image_list, mask_list, num_seg, size, compactness,
                  dataloader, data_augmentation=True, coeff=None,
                    ignore_phase=False, fully_conneted=False):
        self.image_list = image_list
        self.mask_list = mask_list
        self.fully_connected = fully_conneted
        self.resize_mask = ResizeMask(size)
        self.num_seg = num_seg
        
        if dataloader == 'SPGFastFFT':
            totensor = ToTensorSPFFT(num_seg, compactness, coeff, ignore_phase, fully_conneted)
        else:
            totensor = ToTensorSP(num_seg, compactness, fully_conneted)
        

        self.transform = transforms.Compose([Resize(size),
            # [RandomFlip(0.5),
            #  RandomCrop(size, int(size*1.14)),
             totensor])
        if not data_augmentation:
            self.transform = transforms.Compose([Resize(size), totensor])


        self.data_augmentation = data_augmentation

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, item):
        img_name = self.image_list[item]
        mask_name = self.mask_list[item]

        sp_file_name_features = self.image_list[item].split('.')[0].split('/')[-1]+'_features.npy'

        sp_file_name_seq_mask = self.image_list[item].split('.')[0].split('/')[-1]+'_seq_mask.npy'
        sp_file_name_segments = self.image_list[item].split('.')[0].split('/')[-1]+'_segments.npy'




        sp_file_path_features = os.path.join(str(Path(self.image_list[item]).parents[1]),'PTH',sp_file_name_features )
        sp_file_path_seq_mask = os.path.join(str(Path(self.image_list[item]).parents[1]),'PTH',sp_file_name_seq_mask )
        sp_file_path_segments = os.path.join(str(Path(self.image_list[item]).parents[1]),'PTH',sp_file_name_segments )


        mask = Image.open(mask_name)
        mask = mask.convert('L')
        
        if os.path.exists(sp_file_path_features):
            a = time.time()
            features = np.load(sp_file_path_features)
            seq_mask = np.load(sp_file_path_seq_mask)
            segments = np.load(sp_file_path_segments)
            
            mask = self.resize_mask(mask)
            
            if self.fully_connected:
                neighbor_array = np.ones([self.num_seg, self.num_seg])
            else:
                vs_right = np.vstack([segments[:,:-1].ravel(), segments[:,1:].ravel()])
                vs_below = np.vstack([segments[:-1,:].ravel(), segments[1:,:].ravel()])
                vs_diagonal_r = np.vstack([segments[:-1,:-1].ravel(), segments[1:,1:].ravel()])
                vs_diagonal_l = np.vstack([segments[1:,:-1].ravel(), segments[:-1,1:].ravel()])
                bneighbors = np.unique(np.hstack([vs_right, vs_below, vs_diagonal_r, vs_diagonal_l]), axis=1)
                neighbor_array = np.zeros([self.num_seg, self.num_seg])
                neighbor_array[bneighbors[0]-1, bneighbors[1]-1] = 1
                neighbor_array[bneighbors[1]-1, bneighbors[0]-1] = 1
            
            b= time.time()
            

            features_centroids = features[:, :2]
            a = time.time()
            spatial_distances = euclidean_distances(features_centroids, features_centroids)
            b = time.time()
          
            
            
            sample = (torch.tensor(features[:, 2:]).float(), torch.tensor(neighbor_array).float(), torch.tensor(spatial_distances).float().unsqueeze(2), 
                       torch.tensor(seq_mask), torch.tensor(segments), mask, self.image_list[item])

        else:
            os.makedirs(os.path.join(str(Path(self.image_list[item]).parents[1]),'PTH'), exist_ok=True)
            img = Image.open(img_name)
            img = img.convert('RGB')

            sample = {'image': img, 'mask': mask}

            sample = self.transform(sample)
            np.save(sp_file_path_features, sample[0])
            np.save(sp_file_path_seq_mask, sample[1])
            np.save(sp_file_path_segments, sample[2])

            segments = sample[2]
            features = sample[0]

            
            if self.fully_connected:
                neighbor_array = np.ones([self.num_seg, self.num_seg])
            else:
                vs_right = np.vstack([segments[:,:-1].ravel(), segments[:,1:].ravel()])
                vs_below = np.vstack([segments[:-1,:].ravel(), segments[1:,:].ravel()])
                vs_diagonal_r = np.vstack([segments[:-1,:-1].ravel(), segments[1:,1:].ravel()])
                vs_diagonal_l = np.vstack([segments[1:,:-1].ravel(), segments[:-1,1:].ravel()])
                bneighbors = np.unique(np.hstack([vs_right, vs_below, vs_diagonal_r, vs_diagonal_l]), axis=1)
                neighbor_array = np.zeros([self.num_seg, self.num_seg])
                neighbor_array[bneighbors[0]-1, bneighbors[1]-1] = 1
                neighbor_array[bneighbors[1]-1, bneighbors[0]-1] = 1
            edge_index = np.array(np.nonzero(neighbor_array))


            features_centroids = features[:, :2]
            spatial_distances = euclidean_distances(features_centroids, features_centroids)
            spatial_distances = spatial_distances[edge_index]
            
        
            edge_features = np.expand_dims(spatial_distances, axis=1)

            sample = (torch.tensor(features[:, 2:]).float(), torch.tensor(neighbor_array).float(), torch.tensor(spatial_distances).float().unsqueeze(2), 
                       torch.tensor(sample[1]), torch.tensor(sample[2]), sample[3], self.image_list[item])

            
        return sample



class SPGFastDataModule(pl.LightningDataModule):

    def __init__(self, **kwargs):
        super().__init__()

        self.train_dir = kwargs.get('dataset_tr')
        self.test_dir = kwargs.get('dataset_test')
        self.batch_size = kwargs.get('batch_size')
        self.num_workers = kwargs.get('num_workers', 0)
        self.num_seg = kwargs.get('num_seg', 600)
        self.res = kwargs.get('size')
        self.dataloader = kwargs.get('dataloader')
        self.coeff = kwargs.get('coeff')
        self.compactness = kwargs.get('compactness')
        self.ignore_phase = kwargs.get('ignore_phase')
        self.fully_connected = kwargs.get('fully_connected', False)
        
        self.image_list = np.array(sorted([os.path.join('{}/Image'.format(self.train_dir), f) for f in os.listdir('{}/Image'.format(self.train_dir))]))
        self.mask_list = np.array(sorted([os.path.join('{}/Mask'.format(self.train_dir), f) for f in os.listdir('{}/Mask'.format(self.train_dir))]))

        indices = np.array(list(range(len(self.image_list))))
        np.random.shuffle(indices)
        
        self.val_image_list = self.image_list[indices[int(len(self.image_list)*0.85):]]
        self.val_mask_list = self.mask_list[indices[int(len(self.mask_list)*0.85):]]
    
        self.tr_image_list = self.image_list[indices[:int(len(self.image_list)*0.85)]]
        self.tr_mask_list = self.mask_list[indices[:int(len(self.mask_list)*0.85)]]

        self.test_image_list = sorted([os.path.join('{}/Image'.format(self.test_dir), f) for f in os.listdir('{}/Image'.format(self.test_dir))])
        self.test_mask_list = sorted([os.path.join('{}/Mask'.format(self.test_dir), f) for f in os.listdir('{}/Mask'.format(self.test_dir))])

        
    def train_dataloader(self):
        data_train = SPDataset(self.tr_image_list, self.tr_mask_list, self.num_seg,
                                self.res, self.compactness, self.dataloader, True,
                                  self.coeff, self.ignore_phase, self.fully_connected)
        return DataLoader(
                data_train, batch_size=self.batch_size, 
                num_workers=self.num_workers, shuffle=True, pin_memory=False)

    def val_dataloader(self):
        data_val = SPDataset(self.val_image_list, self.val_mask_list, self.num_seg,
                              self.res, self.compactness, self.dataloader,False,
                                self.coeff, self.ignore_phase, self.fully_connected)
        data_test = SPDataset(self.test_image_list, self.test_mask_list, self.num_seg,
                               self.res,  self.compactness, self.dataloader,False, 
                               self.coeff, self.ignore_phase, self.fully_connected)
        val_dataloader = DataLoader(
                data_val, batch_size=self.batch_size, 
                num_workers=self.num_workers, pin_memory=False)
        test_dataloader = DataLoader(
                data_test, batch_size=self.batch_size, 
                num_workers=self.num_workers, pin_memory=False)
        return [val_dataloader, test_dataloader]

    def test_dataloader(self):
        data_test = SPDataset(self.test_image_list, self.test_mask_list, self.num_seg,
                               self.res,  self.compactness, self.dataloader, False,
                                 self.coeff, self.ignore_phase, self.fully_connected)
        return DataLoader(
                data_test, batch_size=self.batch_size, 
                num_workers=self.num_workers, pin_memory=False)



