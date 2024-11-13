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
from skimage import color
import pytorch_lightning as pl
from fast_slic.avx2 import SlicAvx2
from dataset.constants import *
import matplotlib.pyplot as plt
from scipy import sparse as sp
from scipy.spatial.distance import pdist, squareform
from dataset.attributes import *
from torch.utils.data import DataLoader
from dataset.randaugment import RandAugment
from pathlib import Path
from tqdm import tqdm
from dataset.fft_transform import *

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
    def __init__(self, num_seg, compactness, coeff, size, ignore_phase):
        self.tensor = transforms.ToTensor()
        self.num_seg = num_seg
        self.coeff = coeff
        self.compactness = compactness
        self.ignore_phase = ignore_phase
        resample_points = int(((size**2)//num_seg)**0.5)*4
        self.resample_points = resample_points
        
        def fourier_descriptors(region):
            region = (region*255).astype(np.uint8)
            contour, hierarchy = cv2.findContours(region, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            points = contour[0][:, 0, :]
            xi, yi = resample_2d(points, resample_points)
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


    def __call__(self, sample):
        img, mask = sample['image'], sample['mask']
        img_gray = np.array(img.convert('L'))
        img_np = np.array(img)
        mask_np = np.array(mask)/255.
        img_size = img_np.shape

        # img_np = np.ascontiguousarray(np.transpose(img.cpu().numpy()*255, (1, 2, 0))).astype(np.uint8)
            
        # slic = SlicAvx2(num_components=self.num_seg, compactness=self.compactness)
        # segments = slic.iterate(img_np)+1
        segments = slic(img_np, n_segments=self.num_seg,
            compactness=self.compactness,
            max_num_iter=10,
            convert2lab=True,
            enforce_connectivity=False,
            slic_zero=False,
            min_size_factor=0,)

        # plt.imshow(mark_boundaries(img_np, segments))
        # plt.show()

        vs_right = np.vstack([segments[:,:-1].ravel(), segments[:,1:].ravel()])
        vs_below = np.vstack([segments[:-1,:].ravel(), segments[1:,:].ravel()])
        bneighbors, counts = np.unique(np.hstack([vs_right, vs_below]), axis=1, return_counts=True)
        
        
        edge_attr = np.zeros([self.num_seg, self.num_seg])
        for i in range(bneighbors.shape[1]):
            if bneighbors[0,i] != bneighbors[1,i]:
                edge_attr[bneighbors[0,i]-1, bneighbors[1,i]-1] = counts[i]
                
        lbp_np = local_binary_pattern(img_gray, 8, 1, method='uniform')
        regions_lbp = regionprops_table(segments, intensity_image=lbp_np, extra_properties=[self.lbp])

        regions = regionprops_table(segments, intensity_image=img_np, properties=('label', 'centroid', 'intensity_mean',
                                                                                    'coords'), extra_properties=[image_stdev, self.fourier_descriptors])#, polarize])
        
        seq_len = len(regions['label'])
        seq_mask = np.zeros([self.num_seg])
        label = regions['label']
        features = np.zeros([self.num_seg, 8+(self.coeff)*2+10])
        
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

        return features, seq_mask, segments, self.tensor(mask), img_np, edge_attr
    
class ToTensorSP(object):
    def __init__(self, num_seg, compactness):
        self.tensor = transforms.ToTensor()
        self.num_seg = num_seg
        self.compactness = compactness
        

    def __call__(self, sample):
        img, mask = sample['image'], sample['mask']
        img_np = np.array(img)
        mask_np = np.array(mask)/255.
        img_size = img_np.shape

        # img_np = np.ascontiguousarray(np.transpose(img.cpu().numpy()*255, (1, 2, 0))).astype(np.uint8)
            
        # slic = SlicAvx2(num_components=self.num_seg, compactness=self.compactness)
        # segments = slic.iterate(img_np)+1
        segments = slic(img_np, n_segments=self.num_seg,
            compactness=self.compactness,
            max_num_iter=10,
            convert2lab=True,
            enforce_connectivity=False,
            slic_zero=False)
        
     
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


   

        return features, seq_mask, segments, self.tensor(mask), img_np, None

class SPDatasetExport(data.Dataset):
    def __init__(self, image_list, mask_list, num_seg, size, compactness,
                  dataloader,  coeff=None,
                    ignore_phase=False):
        self.image_list = image_list
        self.mask_list = mask_list
        
        self.resize_mask = ResizeMask(size)
        
    
        self.num_seg = num_seg
        self.dataloader = dataloader
        self.size = size
        self.coeff = coeff
        
        if dataloader == 'SPFFFT' or dataloader == 'SPFRS':
            totensor = ToTensorSPFFT(num_seg, compactness, coeff, size, ignore_phase)
        else:
            totensor = ToTensorSP(num_seg, compactness)
        # totensor = ToTensorSPFFT(num_seg, compactness, coeff, ignore_phase, fully_conneted)
        self.transform = transforms.Compose([Resize(size),
             totensor])

        os.makedirs(os.path.join(str(Path(self.image_list[0]).parents[1]), dataloader), exist_ok=True)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, item):
        image = self.image_list[item]
        mask = self.mask_list[item]

        sp_file_name_features = image.split('/')[-1].split('.')[0]+'_features.npy'
        sp_file_name_edge_index = image.split('/')[-1].split('.')[0]+'_edge_index.npy'
        sp_file_name_edge_attr = image.split('/')[-1].split('.')[0]+'_edge_attr.npy'
        sp_file_name_seq_mask = image.split('/')[-1].split('.')[0]+'_seq_mask.npy'
        sp_file_name_segments = image.split('/')[-1].split('.')[0]+'_segments.npy'
        sp_file_name_mask = image.split('/')[-1].split('.')[0]+'_mask.npy'

        sp_file_path_features = os.path.join(str(Path(image).parents[1]),self.dataloader,sp_file_name_features )
        sp_file_path_edge_index = os.path.join(str(Path(image).parents[1]),self.dataloader,sp_file_name_edge_index )
        sp_file_path_edge_attr = os.path.join(str(Path(image).parents[1]),self.dataloader,sp_file_name_edge_attr )
        sp_file_path_seq_mask = os.path.join(str(Path(image).parents[1]),self.dataloader,sp_file_name_seq_mask )
        sp_file_path_segments = os.path.join(str(Path(image).parents[1]),self.dataloader,sp_file_name_segments )
        sp_file_path_mask = os.path.join(str(Path(image).parents[1]),self.dataloader,sp_file_name_mask )
        if os.path.exists(sp_file_path_features):
            return torch.empty(0)
    
        img = Image.open(image)
        img = img.convert('RGB')
        mask = Image.open(mask)
        mask = mask.convert('L')

        sample = {'image': img, 'mask': mask}
        mask = self.resize_mask(mask)
        mask = (mask > 0.5).float()
        sample = self.transform(sample)
        np.save(sp_file_path_features, sample[0])
        np.save(sp_file_path_seq_mask, sample[1])
        np.save(sp_file_path_segments, sample[2])
        np.save(sp_file_path_mask, mask.detach().cpu().numpy())


        return torch.empty(0)

class SPDataset(data.Dataset):
    def __init__(self, image_list, mask_list, num_seg, size, 
                  dataloader, data_augmentation=True, coeff=None):
        self.image_list = image_list
        self.mask_list = mask_list
        self.resize_mask = ResizeMask(size)
        self.num_seg = num_seg
        self.dataloader = dataloader
        self.size = size
        self.coeff = coeff
        self.data_augmentation = data_augmentation
        self.resample_points = int(((size**2)//num_seg)**0.5)*4
            

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, item):
        
        sp_file_name_features = self.image_list[item].split('/')[-1].split('.')[0]+'_features.npy'
        sp_file_name_edge_attr = self.image_list[item].split('/')[-1].split('.')[0]+'_edge_attr.npy.npz'
        sp_file_name_seq_mask = self.image_list[item].split('/')[-1].split('.')[0]+'_seq_mask.npy'
        sp_file_name_segments = self.image_list[item].split('/')[-1].split('.')[0]+'_segments.npy'
        sp_file_name_mask = self.image_list[item].split('/')[-1].split('.')[0]+'_mask.npy'




        sp_file_path_features = os.path.join(str(Path(self.image_list[item]).parents[1]),self.dataloader,sp_file_name_features )
        sp_file_path_edge_attr = os.path.join(str(Path(self.image_list[item]).parents[1]),self.dataloader,sp_file_name_edge_attr )
        sp_file_path_seq_mask = os.path.join(str(Path(self.image_list[item]).parents[1]),self.dataloader,sp_file_name_seq_mask )
        sp_file_path_segments = os.path.join(str(Path(self.image_list[item]).parents[1]),self.dataloader,sp_file_name_segments )
        sp_file_path_mask = os.path.join(str(Path(self.image_list[item]).parents[1]),self.dataloader,sp_file_name_mask)           
        
        features = np.load(sp_file_path_features)
        seq_mask = np.load(sp_file_path_seq_mask)
        segments = np.load(sp_file_path_segments)
        mask = np.load(sp_file_path_mask)
        
        
        
        if self.data_augmentation:
            features, seq_mask = horizontal_flip(features, self.coeff, 0.5, self.size, (int(self.num_seg**0.5), int(self.num_seg**0.5)), seq_mask)
            features = rotate(features, self.coeff, 15, 0.5, (self.size, self.size))
            

        features = torch.tensor(features).float()
        
        if self.data_augmentation:
            randaug = RandAugment(5)
            res = int(self.num_seg**0.5)
            color_space = features[:, 2:5].reshape(res, res, 3).permute(2, 0, 1)
            if np.random.random() < 0.5:
                color_space = (color_space*255).to(torch.uint8)
                color_space, _ = randaug(color_space)
                color_space = color_space.float()
                color_space /= 255.
            # plt.imshow(color_space.permute(1, 2, 0).detach().cpu().numpy())
            # plt.show()
            color_space = color_space.reshape(3, self.num_seg).permute(1, 0)
            
            features[:, 2:5] = color_space
    
        return {'features': features, 'seq_mask': torch.tensor(seq_mask),
                 'segments': torch.tensor(segments), 'mask': mask, 
                   'file_name':self.image_list[item]}
    

class SPOGMaskDataset(data.Dataset):
    def __init__(self, image_list, mask_list, num_seg, size, 
                  dataloader, data_augmentation=True, coeff=None):
        self.image_list = image_list
        self.mask_list = mask_list
        self.resize_mask = ResizeMask(size)
        self.num_seg = num_seg
        self.dataloader = dataloader
        self.size = size
        self.coeff = coeff
        self.data_augmentation = data_augmentation
        self.resample_points = int(((size**2)//num_seg)**0.5)*4
            

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, item):
        
        sp_file_name_features = self.image_list[item].split('/')[-1].split('.')[0]+'_features.npy'
        sp_file_name_edge_attr = self.image_list[item].split('/')[-1].split('.')[0]+'_edge_attr.npy.npz'
        sp_file_name_seq_mask = self.image_list[item].split('/')[-1].split('.')[0]+'_seq_mask.npy'
        sp_file_name_segments = self.image_list[item].split('/')[-1].split('.')[0]+'_segments.npy'
        sp_file_name_mask = self.image_list[item].split('/')[-1].split('.')[0]+'_mask.npy'




        sp_file_path_features = os.path.join(str(Path(self.image_list[item]).parents[1]),self.dataloader,sp_file_name_features )
        sp_file_path_edge_attr = os.path.join(str(Path(self.image_list[item]).parents[1]),self.dataloader,sp_file_name_edge_attr )
        sp_file_path_seq_mask = os.path.join(str(Path(self.image_list[item]).parents[1]),self.dataloader,sp_file_name_seq_mask )
        sp_file_path_segments = os.path.join(str(Path(self.image_list[item]).parents[1]),self.dataloader,sp_file_name_segments )
        sp_file_path_mask = os.path.join(str(Path(self.image_list[item]).parents[1]),self.dataloader,sp_file_name_mask)           
        
        features = np.load(sp_file_path_features)
        seq_mask = np.load(sp_file_path_seq_mask)
        segments = np.load(sp_file_path_segments)
        mask =  Image.open(self.mask_list[item])
        mask = torch.tensor(np.array(mask.convert('L')))/255.
 
        mask = (mask > 0.5).float().unsqueeze(0)
        
        
        if self.data_augmentation:
            features, seq_mask = horizontal_flip(features, self.coeff, 0.5, self.size, (int(self.num_seg**0.5), int(self.num_seg**0.5)), seq_mask)
            features = rotate(features, self.coeff, 15, 0.5, (self.size, self.size))
            

        features = torch.tensor(features).float()
        
        if self.data_augmentation:
            randaug = RandAugment(5)
            res = int(self.num_seg**0.5)
            color_space = features[:, 2:5].reshape(res, res, 3).permute(2, 0, 1)
            if np.random.random() < 0.5:
                color_space = (color_space*255).to(torch.uint8)
                color_space, _ = randaug(color_space)
                color_space = color_space.float()
                color_space /= 255.
            # plt.imshow(color_space.permute(1, 2, 0).detach().cpu().numpy())
            # plt.show()
            color_space = color_space.reshape(3, self.num_seg).permute(1, 0)
            
            features[:, 2:5] = color_space
    
        return {'features': features, 'seq_mask': torch.tensor(seq_mask),
                 'segments': torch.tensor(segments), 'mask': mask, 
                   'file_name':self.image_list[item]}



class SPFDataModule(pl.LightningDataModule):

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
        self.debug = kwargs.get('debug', False)
        self.skip_train = kwargs.get('skip_train')
        
        if not self.skip_train:
            self.image_list = np.array(sorted([os.path.join(os.path.join(self.train_dir, 'Image'), f) for f in os.listdir(os.path.join(self.train_dir, 'Image'))]))
            self.mask_list = np.array(sorted([os.path.join(os.path.join(self.train_dir, 'Mask'), f) for f in os.listdir(os.path.join(self.train_dir, 'Mask'))]))

            indices = np.array(list(range(len(self.image_list))))
            np.random.shuffle(indices)
            
            self.val_image_list = self.image_list[indices[int(len(self.image_list)*0.95):]]
            self.val_mask_list = self.mask_list[indices[int(len(self.mask_list)*0.95):]]
        
            self.tr_image_list = self.image_list[indices[:int(len(self.image_list)*0.95)]]
            self.tr_mask_list = self.mask_list[indices[:int(len(self.mask_list)*0.95)]]
            if self.debug:
                self.val_image_list = self.val_image_list[:100]
                self.val_mask_list = self.val_mask_list[:100]

                self.tr_image_list = self.tr_image_list[:100]
                self.tr_mask_list = self.tr_mask_list[:100]

            dummy_tr = SPDatasetExport(self.tr_image_list, self.tr_mask_list, self.num_seg,
                                self.res, self.compactness, self.dataloader,
                                  self.coeff, self.ignore_phase)
            dummy_tr_loader = DataLoader(
                    dummy_tr, batch_size=1, 
                    num_workers=self.num_workers, shuffle=False, pin_memory=False)
            
            for batch in tqdm(dummy_tr_loader):
                pass

            del dummy_tr, dummy_tr_loader

            dummy_val = SPDatasetExport(self.val_image_list, self.val_mask_list, self.num_seg,
                                self.res, self.compactness, self.dataloader, 
                                    self.coeff, self.ignore_phase)
            
            dummy_val_loader = DataLoader(
                dummy_val, batch_size=1, 
                num_workers=self.num_workers, pin_memory=False)
            
            for batch in tqdm(dummy_val_loader):
                pass

            del dummy_val, dummy_val_loader

        self.test_image_list = sorted([os.path.join(os.path.join(self.test_dir, 'Image'), f) for f in os.listdir(os.path.join(self.test_dir, 'Image'))])
        self.test_mask_list = sorted([os.path.join(os.path.join(self.test_dir, 'Mask'), f) for f in os.listdir(os.path.join(self.test_dir, 'Mask'))])

        
        if self.debug:
            self.test_image_list = self.test_image_list[:100]
            self.test_mask_list = self.test_mask_list[:100]

       
        dummy_test = SPDatasetExport(self.test_image_list, self.test_mask_list, self.num_seg,
                               self.res,  self.compactness, self.dataloader, 
                               self.coeff, self.ignore_phase)
        
        dummy_test_loader = DataLoader(
                dummy_test, batch_size=1, 
                num_workers=self.num_workers, pin_memory=False)
        
        

        for batch in tqdm(dummy_test_loader):
            pass

        del dummy_test, dummy_test_loader, batch
           

        
    def train_dataloader(self):
        data_train = SPDataset(self.tr_image_list, self.tr_mask_list, self.num_seg,
                                self.res, self.dataloader, True,
                                  self.coeff)
        return DataLoader(
                data_train, batch_size=self.batch_size, 
                num_workers=self.num_workers, shuffle=True, pin_memory=True, drop_last=True)

    def val_dataloader(self):
        data_val = SPDataset(self.val_image_list, self.val_mask_list, self.num_seg,
                              self.res, self.dataloader, False,
                                self.coeff)
        data_test = SPDataset(self.test_image_list, self.test_mask_list, self.num_seg,
                               self.res,  self.dataloader, False, 
                               self.coeff)
        val_dataloader = DataLoader(
                data_val, batch_size=self.batch_size, 
                num_workers=self.num_workers, pin_memory=True)
        test_dataloader = DataLoader(
                data_test, batch_size=self.batch_size, 
                num_workers=self.num_workers, pin_memory=True)
        return [val_dataloader, test_dataloader]

    def test_dataloader(self):
        data_test = SPOGMaskDataset(self.test_image_list, self.test_mask_list, self.num_seg,
                               self.res, self.dataloader, False,
                                 self.coeff)
        return DataLoader(
                data_test, batch_size=self.batch_size, 
                num_workers=self.num_workers, pin_memory=True)
    


class SPFRSDataModule(pl.LightningDataModule):

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
        self.debug = kwargs.get('debug', False)
        self.skip_train = kwargs.get('skip_train')
        
        if not self.skip_train:
            self.image_list = np.array(sorted([os.path.join(os.path.join(self.train_dir, 'Image'), f) for f in os.listdir(os.path.join(self.train_dir, 'Image'))]))
            self.mask_list = np.array(sorted([os.path.join(os.path.join(self.train_dir, 'Mask'), f) for f in os.listdir(os.path.join(self.train_dir, 'Mask'))]))

            self.tr_image_list = self.image_list
            self.tr_mask_list = self.mask_list
            if self.debug:
 

                self.tr_image_list = self.tr_image_list[:100]
                self.tr_mask_list = self.tr_mask_list[:100]

            dummy_tr = SPDatasetExport(self.tr_image_list, self.tr_mask_list, self.num_seg,
                                self.res, self.compactness, self.dataloader,
                                  self.coeff, self.ignore_phase)
            dummy_tr_loader = DataLoader(
                    dummy_tr, batch_size=1, 
                    num_workers=self.num_workers, shuffle=False, pin_memory=False)
            
            for batch in tqdm(dummy_tr_loader):
                pass

            del dummy_tr, dummy_tr_loader

        

        self.test_image_list = sorted([os.path.join(os.path.join(self.test_dir, 'Image'), f) for f in os.listdir(os.path.join(self.test_dir, 'Image'))])
        self.test_mask_list = sorted([os.path.join(os.path.join(self.test_dir, 'Mask'), f) for f in os.listdir(os.path.join(self.test_dir, 'Mask'))])

        
        if self.debug:
            self.test_image_list = self.test_image_list[:100]
            self.test_mask_list = self.test_mask_list[:100]

       
        dummy_test = SPDatasetExport(self.test_image_list, self.test_mask_list, self.num_seg,
                               self.res,  self.compactness, self.dataloader, 
                               self.coeff, self.ignore_phase)
        
        dummy_test_loader = DataLoader(
                dummy_test, batch_size=1, 
                num_workers=self.num_workers, pin_memory=False)
        
        

        for batch in tqdm(dummy_test_loader):
            pass

        del dummy_test, dummy_test_loader, batch
           

        
    def train_dataloader(self):
        data_train = SPDataset(self.tr_image_list, self.tr_mask_list, self.num_seg,
                                self.res, self.dataloader, True,
                                  self.coeff)
        return DataLoader(
                data_train, batch_size=self.batch_size, 
                num_workers=self.num_workers, shuffle=True, pin_memory=True, drop_last=True)

    def val_dataloader(self):
        data_val = SPDataset(self.test_image_list, self.test_mask_list, self.num_seg,
                               self.res,  self.dataloader, False, 
                               self.coeff)
        data_test = SPOGMaskDataset(self.test_image_list, self.test_mask_list, self.num_seg,
                               self.res,  self.dataloader, False, 
                               self.coeff)
        val_dataloader = DataLoader(
                data_val, batch_size=self.batch_size, 
                num_workers=self.num_workers, pin_memory=True)
        test_dataloader = DataLoader(
                data_test, batch_size=1, 
                num_workers=self.num_workers, pin_memory=True)
        return [val_dataloader, test_dataloader]

    def test_dataloader(self):
        data_test = SPOGMaskDataset(self.test_image_list, self.test_mask_list, self.num_seg,
                               self.res, self.dataloader, False,
                                 self.coeff)
        return DataLoader(
                data_test, batch_size=1, 
                num_workers=self.num_workers, pin_memory=True)