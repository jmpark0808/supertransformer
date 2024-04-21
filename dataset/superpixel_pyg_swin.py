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
from tqdm import tqdm
import scipy
from torch_geometric.utils.convert import from_scipy_sparse_matrix
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
    def __init__(self, num_seg, compactness, coeff, ignore_phase):
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
            slic_zero=False)

        # plt.imshow(mark_boundaries(img_np, segments))
        # plt.show()

        # vs_right = np.vstack([segments[:,:-1].ravel(), segments[:,1:].ravel()])
        # vs_below = np.vstack([segments[:-1,:].ravel(), segments[1:,:].ravel()])
        # vs_diagonal_r = np.vstack([segments[:-1,:-1].ravel(), segments[1:,1:].ravel()])
        # vs_diagonal_l = np.vstack([segments[1:,:-1].ravel(), segments[:-1,1:].ravel()])
        # bneighbors = np.unique(np.hstack([vs_right, vs_below, vs_diagonal_r, vs_diagonal_l]), axis=1)
    
        lbp_np = local_binary_pattern(img_gray, 8, 1, method='uniform')
        regions_lbp = regionprops_table(segments, intensity_image=lbp_np, extra_properties=[self.lbp])

        regions = regionprops_table(segments, intensity_image=img_np, properties=('label', 'centroid', 'intensity_mean',
                                                                                    'coords'), extra_properties=[image_stdev, self.fourier_descriptors])#, polarize])

        seq_len = len(regions['label'])
        seq_mask = np.zeros([self.num_seg])
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
        segments = slic.iterate(img_np)+1
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

class SPDatasetExport(data.Dataset):
    def __init__(self, image_list, mask_list, num_seg, size, compactness, window_size, dataloader,
                  coeff=None, ignore_phase=False, sigma=None):
        self.image_list = image_list
        self.mask_list = mask_list
        self.resize_mask = ResizeMask(size)
        self.num_seg = num_seg
        self.dataloader = dataloader
        self.sigma = None
        self.size = size
        self.adj_list = {}
        self.window_size = window_size
        
        
        totensor = ToTensorSPFFT(num_seg, compactness, coeff, ignore_phase)
       
        

        self.transform = transforms.Compose([Resize(size),
             totensor])
        

        os.makedirs(os.path.join(str(Path(self.image_list[0]).parents[1]),self.dataloader), exist_ok=True)

        node_index = torch.arange(1024).reshape(1, 32, 32).float().cuda()
        local_unfold = torch.nn.Unfold(self.window_size, stride=self.window_size)
        dilated_unfold = torch.nn.Unfold((32//window_size), dilation=window_size)
        

        local_index = local_unfold(node_index)
        shifted_index = local_unfold(torch.roll(node_index, shifts=(-self.window_size//2, -self.window_size//2), dims=(1, 2)))
        dilated_index = dilated_unfold(node_index)

        all_local_indices = []
        all_shifted_indices = []
        all_dilated_indices = []
        for i in range(local_index.shape[1]):
            x, y = torch.meshgrid(local_index[:, i], local_index[:, i])
            all_local_indices.append(torch.stack((x, y), dim=0).reshape(2, -1))
            x, y = torch.meshgrid(shifted_index[:, i], shifted_index[:, i])
            all_shifted_indices.append(torch.stack((x, y), dim=0).reshape(2, -1))
        for i in range(dilated_index.shape[1]):
            x, y = torch.meshgrid(dilated_index[:, i], dilated_index[:, i])
            all_dilated_indices.append(torch.stack((x, y), dim=0).reshape(2, -1))
            
        all_local_indices = torch.cat(all_local_indices, dim=1)
        all_shifted_indices = torch.cat(all_shifted_indices, dim=1)
        all_dilated_indices = torch.cat(all_dilated_indices, dim=1)

        edge_indices = torch.cat((all_local_indices, all_shifted_indices, all_dilated_indices), dim=1).long().detach().cpu().numpy()
        self.edge_indices = edge_indices

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, item):
        image = self.image_list[item]
        mask = self.mask_list[item]
        
        

        sp_file_name_features = image.split('/')[-1].split('.')[0]+'_features.npy'
        sp_file_name_edge_features = image.split('/')[-1].split('.')[0]+'_edge_features.npy'
        sp_file_name_seq_mask = image.split('/')[-1].split('.')[0]+'_seq_mask.npy'
        sp_file_name_segments = image.split('/')[-1].split('.')[0]+'_segments.npy'
        sp_file_name_mask = image.split('/')[-1].split('.')[0]+'_mask.npy'

        sp_file_path_features = os.path.join(str(Path(image).parents[1]),self.dataloader,sp_file_name_features )
        sp_file_path_edge_features = os.path.join(str(Path(image).parents[1]),self.dataloader,sp_file_name_edge_features )
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

        segments = sample[2]
        features = sample[0]
    
       
        features_centroids = features[:, :2]/self.size
        spatial_distances_x = (features_centroids[:, 0:1] - features_centroids[:, 0:1].T)
        spatial_distances_x = spatial_distances_x[self.edge_indices[0], self.edge_indices[1]]
        spatial_distances_y = (features_centroids[:, 1:2] - features_centroids[:, 1:2].T)
        spatial_distances_y = spatial_distances_y[self.edge_indices[0], self.edge_indices[1]]
        spatial_distances = np.stack((spatial_distances_x, spatial_distances_y), axis=1)
        
        # edge_features = spatial_distances
        np.save(sp_file_path_edge_features, spatial_distances)

        
        return torch.empty(0)
        

class SPDataset(data.Dataset):
    def __init__(self, image_list, mask_list, num_seg, size, 
                  dataloader, window_size,  coeff=None,
                     sigma=None, memory=False):
        self.image_list = image_list
        self.mask_list = mask_list
        self.resize_mask = ResizeMask(size)
        self.num_seg = num_seg
        self.dataloader = dataloader
        self.sigma = sigma
        self.size = size
        self.coeff = coeff
        self.window_size = window_size

        
        node_index = torch.arange(1024).reshape(1, 32, 32).float()
        local_unfold = torch.nn.Unfold(self.window_size, stride=self.window_size)
        dilated_unfold = torch.nn.Unfold((32//window_size), dilation=window_size)
        

        local_index = local_unfold(node_index)
        shifted_index = local_unfold(torch.roll(node_index, shifts=(-self.window_size//2, -self.window_size//2), dims=(1, 2)))
        dilated_index = dilated_unfold(node_index)

        all_local_indices = []
        all_shifted_indices = []
        all_dilated_indices = []
        for i in range(local_index.shape[1]):
            x, y = torch.meshgrid(local_index[:, i], local_index[:, i])
            all_local_indices.append(torch.stack((x, y), dim=0).reshape(2, -1))
            x, y = torch.meshgrid(shifted_index[:, i], shifted_index[:, i])
            all_shifted_indices.append(torch.stack((x, y), dim=0).reshape(2, -1))
        for i in range(dilated_index.shape[1]):
            x, y = torch.meshgrid(dilated_index[:, i], dilated_index[:, i])
            all_dilated_indices.append(torch.stack((x, y), dim=0).reshape(2, -1))
            
        all_local_indices = torch.cat(all_local_indices, dim=1)
        all_shifted_indices = torch.cat(all_shifted_indices, dim=1)
        all_dilated_indices = torch.cat(all_dilated_indices, dim=1)

        edge_indices = torch.cat((all_local_indices, all_shifted_indices, all_dilated_indices), dim=1).long()
        self.edge_indices = edge_indices
        
        self.features = []
        self.edge_features = []
        self.seq_mask = []
        self.segments = []
        self.mask = []
        self.memory = memory
        if memory:

            for item in range(len(self.image_list)):
                sp_file_name_features = self.image_list[item].split('/')[-1].split('.')[0]+'_features.npy'
                sp_file_name_edge_index = self.image_list[item].split('/')[-1].split('.')[0]+'_edge_index.npy'
                sp_file_name_edge_features = self.image_list[item].split('/')[-1].split('.')[0]+'_edge_features.npy'
                sp_file_name_seq_mask = self.image_list[item].split('/')[-1].split('.')[0]+'_seq_mask.npy'
                sp_file_name_segments = self.image_list[item].split('/')[-1].split('.')[0]+'_segments.npy'
                sp_file_name_mask = self.image_list[item].split('/')[-1].split('.')[0]+'_mask.npy'

                sp_file_path_features = os.path.join(str(Path(self.image_list[item]).parents[1]),self.dataloader,sp_file_name_features )
                sp_file_path_edge_index = os.path.join(str(Path(self.image_list[item]).parents[1]),self.dataloader,sp_file_name_edge_index )
                sp_file_path_edge_features = os.path.join(str(Path(self.image_list[item]).parents[1]),self.dataloader,sp_file_name_edge_features )
                sp_file_path_seq_mask = os.path.join(str(Path(self.image_list[item]).parents[1]),self.dataloader,sp_file_name_seq_mask )
                sp_file_path_segments = os.path.join(str(Path(self.image_list[item]).parents[1]),self.dataloader,sp_file_name_segments )
                sp_file_path_mask = os.path.join(str(Path(self.image_list[item]).parents[1]),self.dataloader,sp_file_name_mask)

                features = np.load(sp_file_path_features)
                seq_mask = np.load(sp_file_path_seq_mask)
                segments = np.load(sp_file_path_segments)
                edge_index = self.edge_indices
                mask = np.load(sp_file_path_mask)
                edge_attr = np.load(sp_file_path_edge_features)

                self.features.append(features)
                self.seq_mask.append(seq_mask)
                self.segments.append(segments)
                self.mask.append(mask)
                self.edge_features.append(edge_attr)
            
       
     


    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, item):
        if not self.memory:
            sp_file_name_features = self.image_list[item].split('/')[-1].split('.')[0]+'_features.npy'
            sp_file_name_edge_index = self.image_list[item].split('/')[-1].split('.')[0]+'_edge_index.npy'
            sp_file_name_edge_features = self.image_list[item].split('/')[-1].split('.')[0]+'_edge_features.npy'
            sp_file_name_seq_mask = self.image_list[item].split('/')[-1].split('.')[0]+'_seq_mask.npy'
            sp_file_name_segments = self.image_list[item].split('/')[-1].split('.')[0]+'_segments.npy'
            sp_file_name_mask = self.image_list[item].split('/')[-1].split('.')[0]+'_mask.npy'

            sp_file_path_features = os.path.join(str(Path(self.image_list[item]).parents[1]),self.dataloader,sp_file_name_features )
            sp_file_path_edge_index = os.path.join(str(Path(self.image_list[item]).parents[1]),self.dataloader,sp_file_name_edge_index )
            sp_file_path_edge_features = os.path.join(str(Path(self.image_list[item]).parents[1]),self.dataloader,sp_file_name_edge_features )
            sp_file_path_seq_mask = os.path.join(str(Path(self.image_list[item]).parents[1]),self.dataloader,sp_file_name_seq_mask )
            sp_file_path_segments = os.path.join(str(Path(self.image_list[item]).parents[1]),self.dataloader,sp_file_name_segments )
            sp_file_path_mask = os.path.join(str(Path(self.image_list[item]).parents[1]),self.dataloader,sp_file_name_mask)

            features = np.load(sp_file_path_features)
            seq_mask = np.load(sp_file_path_seq_mask)
            segments = np.load(sp_file_path_segments)
            edge_index = self.edge_indices
            mask = np.load(sp_file_path_mask)
            edge_attr = np.load(sp_file_path_edge_features)
            


            # edge_features = np.load(sp_file_path_edge_features)
            
            
            if self.sigma is not None and self.dataloader == 'SPGSWIN':
                features = horizontal_flip(features, self.coeff, 0.5, self.size)
                gaussian_noise = np.random.normal(1, self.sigma, features.shape)
                features = features*gaussian_noise
            
            sample = (Data(x=torch.tensor(features).float(),
                            edge_index=edge_index,
                                edge_attr=torch.tensor(edge_attr).float()),
                        torch.tensor(seq_mask), torch.tensor(segments), torch.tensor(mask), self.image_list[item])
        else:
            features = self.features[item]
            seq_mask = self.seq_mask[item]
            segments = self.segments[item]
            edge_index = self.edge_indices
            mask = self.mask[item]
            edge_attr = self.edge_features[item]

            if self.sigma is not None and self.dataloader == 'SPGSWIN':
                features = horizontal_flip(features, self.coeff, 0.5, self.size)
                gaussian_noise = np.random.normal(1, self.sigma, features.shape)
                features = features*gaussian_noise
            
            sample = (Data(x=torch.tensor(features).float(),
                            edge_index=edge_index,
                                edge_attr=torch.tensor(edge_attr).float()),
                        torch.tensor(seq_mask), torch.tensor(segments), torch.tensor(mask), self.image_list[item])

    
        return sample



class SPGSWINDataModule(pl.LightningDataModule):

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
        self.sigma = kwargs.get('sigma', None)
        self.dilation = kwargs.get('dilation')
        self.debug = kwargs.get('debug', False)
        self.dilation_mode = kwargs.get('dilation_mode', 0)
        self.window_size = kwargs.get('window_size', 4)
        self.memory = kwargs.get('memory', False)
        
        self.image_list = np.array(sorted([os.path.join(os.path.join(self.train_dir, 'Image'), f) for f in os.listdir(os.path.join(self.train_dir, 'Image'))]))
        self.mask_list = np.array(sorted([os.path.join(os.path.join(self.train_dir, 'Mask'), f) for f in os.listdir(os.path.join(self.train_dir, 'Mask'))]))

        indices = np.array(list(range(len(self.image_list))))
        np.random.shuffle(indices)
        
        self.val_image_list = self.image_list[indices[int(len(self.image_list)*0.85):]]
        self.val_mask_list = self.mask_list[indices[int(len(self.mask_list)*0.85):]]
    
        self.tr_image_list = self.image_list[indices[:int(len(self.image_list)*0.85)]]
        self.tr_mask_list = self.mask_list[indices[:int(len(self.mask_list)*0.85)]]

        self.test_image_list = sorted([os.path.join(os.path.join(self.test_dir, 'Image'), f) for f in os.listdir(os.path.join(self.test_dir, 'Image'))])
        self.test_mask_list = sorted([os.path.join(os.path.join(self.test_dir, 'Mask'), f) for f in os.listdir(os.path.join(self.test_dir, 'Mask'))])

        if self.debug:
            self.val_image_list = self.val_image_list[:100]
            self.val_mask_list = self.val_mask_list[:100]

            self.tr_image_list = self.tr_image_list[:100]
            self.tr_mask_list = self.tr_mask_list[:100]

            self.test_image_list = self.test_image_list[:100]
            self.test_mask_list = self.test_mask_list[:100]

        dummy_tr = SPDatasetExport(self.tr_image_list, self.tr_mask_list, self.num_seg,
                                self.res, self.compactness, self.window_size, self.dataloader,  self.coeff, self.ignore_phase, None)
        dummy_tr_loader = DataLoader(
                dummy_tr, batch_size=self.batch_size, 
                num_workers=self.num_workers, shuffle=False, pin_memory=False)
        
        for batch in tqdm(dummy_tr_loader):
            pass

        del dummy_tr, dummy_tr_loader

        dummy_val = SPDatasetExport(self.val_image_list, self.val_mask_list, self.num_seg,
                              self.res, self.compactness, self.window_size, self.dataloader,
                                self.coeff, self.ignore_phase,  None)
        dummy_test = SPDatasetExport(self.test_image_list, self.test_mask_list, self.num_seg,
                               self.res,  self.compactness, self.window_size, self.dataloader, 
                               self.coeff, self.ignore_phase, None)
        dummy_val_loader = DataLoader(
                dummy_val, batch_size=self.batch_size, 
                num_workers=self.num_workers, pin_memory=False)
        dummy_test_loader = DataLoader(
                dummy_test, batch_size=self.batch_size, 
                num_workers=self.num_workers, pin_memory=False)
        
        for batch in tqdm(dummy_val_loader):
            pass

        del dummy_val, dummy_val_loader

        for batch in tqdm(dummy_test_loader):
            pass

        del dummy_test, dummy_test_loader, batch


    def train_dataloader(self):
        data_train = SPDataset(self.tr_image_list, self.tr_mask_list, self.num_seg,
                                self.res, self.dataloader, self.window_size,
                                  self.coeff,  self.sigma, self.memory)
        return DataLoader(
                data_train, batch_size=self.batch_size, 
                num_workers=self.num_workers, shuffle=True, pin_memory=False)

    def val_dataloader(self):
        data_val = SPDataset(self.val_image_list, self.val_mask_list, self.num_seg,
                              self.res, self.dataloader,self.window_size,
                                self.coeff, None, self.memory)
        data_test = SPDataset(self.test_image_list, self.test_mask_list, self.num_seg,
                               self.res, self.dataloader,self.window_size,
                               self.coeff, None, self.memory)
        val_dataloader = DataLoader(
                data_val, batch_size=self.batch_size, 
                num_workers=self.num_workers, pin_memory=False)
        test_dataloader = DataLoader(
                data_test, batch_size=self.batch_size, 
                num_workers=self.num_workers, pin_memory=False)
        return [val_dataloader, test_dataloader]

    def test_dataloader(self):
        data_test = SPDataset(self.test_image_list, self.test_mask_list, self.num_seg,
                               self.res,  self.dataloader, self.window_size,
                                 self.coeff,None, self.memory)
        return DataLoader(
                data_test, batch_size=self.batch_size, 
                num_workers=self.num_workers, pin_memory=False)