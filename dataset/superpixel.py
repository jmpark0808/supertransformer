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
from torch.utils.data import DataLoader
from fast_slic.avx2 import SlicAvx2
from dataset.constants import *
import matplotlib.pyplot as plt
from scipy import sparse as sp
from scipy.spatial.distance import pdist, squareform
from dataset.attributes import *
from pathlib import Path
from dataset.randaugment import RandAugment
import torch.nn.functional as F



class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        img, mask = sample['image'], sample['mask']
        img = img.resize((self.size, self.size), resample=Image.BILINEAR)
        mask = mask.resize((self.size, self.size), resample=Image.BILINEAR)
        
        return {'image': img, 'mask': mask}
    
class ResizeDownsample(object):
    def __init__(self, size):
        self.size = size
        

    def __call__(self, sample):
        img, mask = sample['image'], sample['mask']
        img, mask = img.resize((self.size, self.size), resample=Image.BILINEAR), mask.resize((self.size, self.size),
                                                                                             resample=Image.BILINEAR)
        
        return {'image': img, 'mask': mask, 'file_name': sample['file_name']}


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

        return {'image': img, 'mask': mask, 'file_name': sample['file_name']}


class RandomFlip(object):
    def __init__(self, prob):
        self.prob = prob
        self.flip = transforms.RandomHorizontalFlip(1.)

    def __call__(self, sample):
        if np.random.random_sample() < self.prob:
            img, mask = sample['image'], sample['mask']
            img = self.flip(img)
            mask = self.flip(mask)
            return {'image': img, 'mask': mask, 'file_name': sample['file_name']}
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
        return {'image': img, 'mask': mask, 'file_name': sample['file_name']}

class RandomColorJitter(object):
    def __init__(self, brightness, contrast, saturation, hue) -> None:
        self.transform = transforms.ColorJitter(brightness, contrast, saturation, hue)

    def __call__(self, sample):
        img, mask = sample['image'], sample['mask']

        img = self.transform(img)

        return {'image': img, 'mask': mask}



class ToTensorSP(object):
    def __init__(self, num_seg, compactness, size):
        self.tensor = transforms.ToTensor()
        self.num_seg = num_seg
        self.compactness = compactness
        xs = torch.arange(0, size).unsqueeze(0).float()
        ys = torch.arange(0, size).unsqueeze(1).float()
        xs = xs.repeat(size, 1)
        ys = ys.repeat(1, size)
        self.coords = torch.stack((xs, ys), 2)

    def __call__(self, sample):
        img, mask = sample['image'], sample['mask']
        img_np = np.array(img)
        


        # slic = SlicAvx2(num_components=self.num_seg, compactness=self.compactness, min_size_factor=0.)
        # segments = slic.iterate(img_np)

        
        # segments = torch.tensor(segments).long()

        # features = np.zeros([self.num_seg, 4])
        # seq_mask = np.zeros([self.num_seg])
        # for i in range(self.num_seg):
        #     where = np.argwhere(segments==i)
        #     area = where.shape[0]
        #     mean_colour = img_np[where[:, 0], where[:, 1], :].mean(0)
        #     centroid = self.coords[where[:, 0], where[:, 1], :].mean(0)


        # label_onehot = F.one_hot(segments, self.num_seg).float()
        segments = slic(img_np, n_segments=self.num_seg,
            compactness=self.compactness,
            max_num_iter=1,
            convert2lab=True,
            enforce_connectivity=False,
            slic_zero=False)-1

        # segments = cuda_slic(img_np, n_segments=self.num_seg, compactness=self.compactness, convert2lab=False, enforce_connectivity=False, )
   
        # vs_right = np.vstack([segments[:,:-1].ravel(), segments[:,1:].ravel()])
        # vs_below = np.vstack([segments[:-1,:].ravel(), segments[1:,:].ravel()])
        # vs_diagonal_r = np.vstack([segments[:-1,:-1].ravel(), segments[1:,1:].ravel()])
        # vs_diagonal_l = np.vstack([segments[1:,:-1].ravel(), segments[:-1,1:].ravel()])
        # bneighbors = np.unique(np.hstack([vs_right, vs_below, vs_diagonal_r, vs_diagonal_l]), axis=1)

        # regions = regionprops_table(segments, intensity_image=img_np, properties=('label', 'centroid'))#, polarize])
        # centers_y = regions['centroid-0']
        # centers_x = regions['centroid-1']
        # plt.scatter(centers_x, centers_y, c='blue', s=30)
        # for ind, (x, y) in enumerate(zip(centers_x, centers_y)):
        #     plt.text(x, y, str(regions['label'][ind]))
        # plt.show()
                    
        # seq_len = len(regions['label'])
        # features = np.zeros([self.num_seg, 5])
        # seq_mask = np.zeros([self.num_seg])
        # label = regions['label']
        # features[label-1, 0] = regions['centroid-0']
        # features[label-1, 1] = regions['centroid-1']
        # features[label-1, 2] = regions['intensity_mean-0']/255.
        # features[label-1, 3] = regions['intensity_mean-1']/255.
        # features[label-1, 4] = regions['intensity_mean-2']/255.


        # for ind, coord in zip(regions['label'], regions['coords']):
        #     seq_mask[ind-1] = 1 if np.sum(mask_np[coord[:, 0], coord[:, 1]])/len(coord[:, 0]) >= 0.5 else 0

        # neighbor_array = np.zeros([self.num_seg, self.num_seg])
        # eye = np.eye(self.num_seg)
        # neighbor_array[bneighbors[0]-1, bneighbors[1]-1] = 1
        # neighbor_array[bneighbors[1]-1, bneighbors[0]-1] = 1
        # neighbor_array -= eye


        # A = neighbor_array.astype(float)
        # N = sp.diags(np.sum(A, axis=0)** -0.5, dtype=float)
        # L = eye - N * A * N

        # # Eigenvectors with numpy
        # EigVal, EigVec = np.linalg.eig(L)
        # idx = EigVal.argsort() # increasing order
        # EigVal, EigVec = EigVal[idx], np.real(EigVec[:,idx])
        # pos_enc = torch.from_numpy(EigVec[:,1:POS_EMBEDDING+1]).float() 

        # histogram_r = np.zeros([self.num_seg, BINS])
        # histogram_g = np.zeros([self.num_seg, BINS])
        # histogram_b = np.zeros([self.num_seg, BINS])
        # for i in range(BINS):
        #     histogram_r[label-1, i] = regions[f'hist-{i}-0']
        #     histogram_g[label-1, i] = regions[f'hist-{i}-1']
        #     histogram_b[label-1, i] = regions[f'hist-{i}-2']

        # histogram_r = histogram_r/np.sum(histogram_r, axis=1, keepdims=True)
        # histogram_g = histogram_g/np.sum(histogram_g, axis=1, keepdims=True)
        # histogram_b = histogram_b/np.sum(histogram_b, axis=1, keepdims=True)
        
        # histogram_r_sq = 1-pdist(histogram_r, lambda u, v: np.sqrt(u*v).sum())
        # histogram_g_sq = 1-pdist(histogram_g, lambda u, v: np.sqrt(u*v).sum())
        # histogram_b_sq = 1-pdist(histogram_b, lambda u, v: np.sqrt(u*v).sum())

        # # spatial_distances = euclidean_distances(features[:, :2], features[:, :2])/np.sqrt(300**2+300**2)
        # spatial_distances_x = (features[:, 0:1] - features[:, 0:1].T)/300.
        # spatial_distances_y = (features[:, 1:2] - features[:, 1:2].T)/300.
        # # ind = np.argsort(distances, axis=1)
        # # neighbor_array = ind <= NUM_NEIGHBOURS
        # # neighbor_array = np.zeros([self.num_seg, self.num_seg])
        
        # edge_features = np.stack((spatial_distances_x, spatial_distances_y, squareform(histogram_r_sq), squareform(histogram_g_sq), squareform(histogram_b_sq)), axis=2)

        # features = torch.zeros([1])
        seq_mask = torch.zeros([1])
        features, seq_mask, segments, mask = self.tensor(img), seq_mask, torch.tensor(segments), self.tensor(mask)
        
        return {'features': features, 'seq_mask': seq_mask, 'segments': segments, 'mask': mask}
    


class ToTensorSPDummy(object):
    def __init__(self, num_seg, compactness, size, ec, sz):
        self.tensor = transforms.ToTensor()
        self.num_seg = num_seg
        self.compactness = compactness
        xs = torch.arange(0, size).unsqueeze(0).float()
        ys = torch.arange(0, size).unsqueeze(1).float()
        xs = xs.repeat(size, 1)
        ys = ys.repeat(1, size)
        self.coords = torch.stack((xs, ys), 2)
        self.ec = ec
        self.sz = sz

    def __call__(self, sample):
        img, mask = sample['image'], sample['mask']
        img_np = np.array(img)
        
        segments = slic(img_np, n_segments=self.num_seg,
            compactness=self.compactness,
            max_num_iter=10,
            convert2lab=True,
            enforce_connectivity=self.ec,
            slic_zero=self.sz)-1

        
        seq_mask = torch.zeros([1])
        features, seq_mask, segments, mask = self.tensor(img), seq_mask, torch.tensor(segments), self.tensor(mask)
        
        return {'features': features, 'seq_mask': seq_mask, 'segments': segments, 'mask': mask}

class ToTensorSPLAP(object):
    def __init__(self, num_seg, compactness):
        self.tensor = transforms.ToTensor()
        self.num_seg = num_seg
        self.compactness = compactness

    def __call__(self, sample):
        img, mask = sample['image'], sample['mask']
        img_np = np.array(img)
        img_size = img_np.shape[1]
        mask_np = np.array(mask)/255.
        segments = slic(img_np, n_segments=self.num_seg,
            compactness=self.compactness,
            max_num_iter=3,
            convert2lab=True,
            enforce_connectivity=False,
            slic_zero=False)
   
        vs_right = np.vstack([segments[:,:-1].ravel(), segments[:,1:].ravel()])
        vs_below = np.vstack([segments[:-1,:].ravel(), segments[1:,:].ravel()])
        vs_diagonal_r = np.vstack([segments[:-1,:-1].ravel(), segments[1:,1:].ravel()])
        vs_diagonal_l = np.vstack([segments[1:,:-1].ravel(), segments[:-1,1:].ravel()])
        bneighbors = np.unique(np.hstack([vs_right, vs_below, vs_diagonal_r, vs_diagonal_l]), axis=1)

        regions = regionprops_table(segments, intensity_image=img_np, properties=('label', 'centroid', 'area', 'intensity_mean',
                                                                                     'coords'), extra_properties=[image_stdev])#, polarize])
                    
        seq_len = len(regions['label'])
        features = np.zeros([self.num_seg, 9])
        seq_mask = np.zeros([self.num_seg])
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


        for ind, coord in zip(regions['label'], regions['coords']):
            seq_mask[ind-1] = 1 if np.sum(mask_np[coord[:, 0], coord[:, 1]])/len(coord[:, 0]) >= 0.5 else 0

        neighbor_array = np.zeros([self.num_seg, self.num_seg])
        eye = np.eye(self.num_seg)
        neighbor_array[bneighbors[0]-1, bneighbors[1]-1] = 1
        neighbor_array[bneighbors[1]-1, bneighbors[0]-1] = 1
        neighbor_array -= eye


        A = neighbor_array.astype(float)
        N = sp.diags(np.sum(A, axis=0).clip(1) ** -0.5, dtype=float)
        L = eye - N * A * N


        # Eigenvectors with numpy
        EigVal, EigVec = np.linalg.eig(L)
        idx = EigVal.argsort() # increasing order
        EigVal, EigVec = EigVal[idx], np.real(EigVec[:,idx])
        pos_enc = torch.from_numpy(EigVec[:,1:]).float() 

     

        features, neighbor_array, seq_mask, segments, mask, img = torch.tensor(features).float(), torch.tensor(neighbor_array).float(), torch.tensor(seq_mask).float(), torch.tensor(segments), self.tensor(mask), self.tensor(img)
        return {'features': features, 'seq_mask': seq_mask, 'segments': segments, 'mask': mask, 'img': img, 'neighbor_array': neighbor_array, 'pos_enc': pos_enc}
 


class ToTensorSPFFT(object):
    def __init__(self, num_seg, compactness, coeff, size, ignore_phase):
        self.tensor = transforms.ToTensor()
        self.num_seg = num_seg
        self.coeff = coeff
        self.compactness = compactness
        self.ignore_phase = ignore_phase
        self.size = size
        
        resample_points = int(((size**2)//num_seg)**0.5)*4
        self.resample_points = resample_points
        
        def fourier_descriptors(region):
            moments = compute_central_moments(region)
            region = (region*255).astype(np.uint8)
            contour, hierarchy = cv2.findContours(region, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            if len(contour)>1:
                merged_contour = merge_contours(contour)

                points = np.array(merged_contour).reshape((-1, 2)).astype(np.int32)

                indices_y = np.argwhere(points[:, 1]==np.min(points[:, 1])) # smallest y
                indices_x = np.argmin(points[indices_y, 0])
                points = np.roll(points, -indices_y[indices_x], axis=0)
            else:
                points = contour[0][:, 0, :]
            xi, yi = resample_2d(points, resample_points)
            contour_array = np.stack((xi, yi), axis=1)


            contour_complex = np.empty(contour_array.shape[:-1], dtype=complex)
            contour_complex.real = contour_array[:, 0]
            contour_complex.imag = contour_array[:, 1]
            fourier_result = np.fft.fft(contour_complex)[1:]


            amp = abs(fourier_result)

            front = math.ceil(self.coeff/2.)
            back = self.coeff-front
            assert (front+back) <= (self.resample_points-1)
            amp = np.concatenate((amp[1:front+1], amp[-back:]), axis=0)
            phase = np.arctan2(fourier_result.imag, fourier_result.real)

            # return np.array(amp)
            return np.concatenate((amp, moments))
        
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
        # fig, ax = plt.subplots(1, 2)
        # ax[0].imshow(img)
        # ax[1].imshow(mask, cmap='gray')
        # plt.show()
        img_gray = np.array(img.convert('L'))
        img_np = np.array(img)
        mask_np = np.array(mask)/255.

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
        features = np.zeros([self.num_seg, 8+(self.coeff)+8+10])
        
        for i in range((self.coeff)+8):
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
            # features[label-1, ind+8+(self.resample_points-1)*2] = regions_lbp[f'lbp-{ind}']
            features[label-1, ind+8+(self.coeff)+8] = regions_lbp[f'lbp-{ind}']

        for ind, coord in zip(regions['label'], regions['coords']):
            seq_mask[ind-1] = np.sum(mask_np[coord[:, 0], coord[:, 1]])/len(coord[:, 0])


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
        features, seq_mask, segments, mask, img = torch.tensor(features).float(),  torch.tensor(seq_mask).float(), torch.tensor(segments), self.tensor(mask), self.tensor(img)
        return {'features': features, 'seq_mask': seq_mask, 'segments': segments, 'mask': mask, 'img': img}
 

class ToTensorSPContour(object):
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
            enforce_connectivity=True,
            slic_zero=False)


        vs_right = np.vstack([segments[:,:-1].ravel(), segments[:,1:].ravel()])
        vs_below = np.vstack([segments[:-1,:].ravel(), segments[1:,:].ravel()])
        vs_diagonal_r = np.vstack([segments[:-1,:-1].ravel(), segments[1:,1:].ravel()])
        vs_diagonal_l = np.vstack([segments[1:,:-1].ravel(), segments[:-1,1:].ravel()])
        bneighbors = np.unique(np.hstack([vs_right, vs_below, vs_diagonal_r, vs_diagonal_l]), axis=1)
    

        regions = regionprops_table(segments, intensity_image=img_np, properties=('label', 'centroid', 'area', 'intensity_mean',
                                                                                    'coords'), extra_properties=[image_stdev, contours_euc])#, polarize])


        seq_len = len(regions['label'])
        features = np.zeros([self.num_seg, 8+(RESAMPLE_POINTS*2)])
        seq_mask = np.zeros([self.num_seg])
        label = regions['label']
        features[label-1, 0] = regions['centroid-0']
        features[label-1, 1] = regions['centroid-1']
        features[label-1, 2] = regions['intensity_mean-0']/255.
        features[label-1, 3] = regions['intensity_mean-1']/255.
        features[label-1, 4] = regions['intensity_mean-2']/255.
        features[label-1, 5] = regions['image_stdev-0']/255.
        features[label-1, 6] = regions['image_stdev-1']/255.
        features[label-1, 7] = regions['image_stdev-2']/255.
        for i in range(RESAMPLE_POINTS):
            features[label-1, 8+i] = regions[f'contours_euc-{i}-0']
            features[label-1, 8+RESAMPLE_POINTS+i] = regions[f'contours_euc-{i}-1']


        for ind, coord in zip(regions['label'], regions['coords']):
            seq_mask[ind-1] = 1 if np.sum(mask_np[coord[:, 0], coord[:, 1]])/len(coord[:, 0]) >= 0.5 else 0

        neighbor_array = np.zeros([self.num_seg, self.num_seg])
        # eye = np.eye(self.num_seg)
        neighbor_array[bneighbors[0]-1, bneighbors[1]-1] = 1
        neighbor_array[bneighbors[1]-1, bneighbors[0]-1] = 1
        # neighbor_array -= eye


        features, neighbor_array, seq_mask, segments, mask, img = torch.tensor(features).float(), torch.tensor(neighbor_array).float(), torch.tensor(seq_mask).float(), torch.tensor(segments), self.tensor(mask), self.tensor(img)

        return {'features': features, 'seq_mask': seq_mask, 'segments': segments, 'mask': mask, 'img': img, 'neighbor_array': neighbor_array, }


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
        # segments = slic.iterate(img_np)+1
    
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
        features[label-1, 2] = regions['intensity_mean-0']
        features[label-1, 3] = regions['intensity_mean-1']
        features[label-1, 4] = regions['intensity_mean-2']
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
    def __init__(self, data_augmentation):
        self.tensor = transforms.ToTensor()
        self.data_augmentation = data_augmentation

    def __call__(self, sample):
        img, mask = sample['image'], sample['mask']
        img, mask = self.tensor(img), self.tensor(mask)
        if self.data_augmentation:
            randaug = RandAugment(5)
            img = (img*255).to(torch.uint8)
            img = randaug(img).float()
            img /= 255.
            
        # Add centroids
        

        # return {'image': img, 'mask': mask}
        return {'features': img, 'seq_mask': mask,
                 'segments': torch.empty(0), 'mask': mask, 
                   'file_name': sample['file_name']}

class SPDataset(data.Dataset):
    def __init__(self, image_list, mask_list, num_seg, size, compactness, data_augmentation=True, dataloader=None, coeff=None, ignore_phase=False):
        self.image_list = image_list
        self.mask_list = mask_list
        
        if dataloader == 'SP':
            totensor = ToTensorSP(num_seg, compactness, size)
        elif dataloader == 'SPFFT' or dataloader == 'SPRS':
            totensor = ToTensorSPFFT(num_seg, compactness, coeff, size, ignore_phase)
        elif dataloader == 'SPLAP':
            totensor = ToTensorSPLAP(num_seg, compactness)
        elif dataloader == 'SPCNN':
            totensor = ToTensorSPCNN(num_seg, compactness)
        elif dataloader == 'SPContour':
            totensor = ToTensorSPContour(num_seg, compactness)
        else:
            raise 'Unrecongized dataloader'

        self.transform = transforms.Compose(
            [RandomFlip(0.5),
             RandomCrop(size, int(size*1.14)),
             RandomAffine(15, 0.1, 0.1),
             RandomColorJitter(0.2, 0.2, 0.2, 0.2),
             Resize(size),
             totensor])
        if not data_augmentation:
            self.transform = transforms.Compose([Resize(size), totensor])

        self.data_augmentation = data_augmentation
        if self.data_augmentation is False:
            os.makedirs(os.path.join(str(Path(self.image_list[0]).parents[1]),'VAL'), exist_ok=True)


    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, item):
        file_name = self.image_list[item].split('/')[-1].split('.')[0]+'.npy'
        file_path = os.path.join(str(Path(self.image_list[item]).parents[1]),'VAL', file_name )

        # if self.data_augmentation is False and os.path.exists(file_path):
        #     sample = np.load(file_path, allow_pickle=True).item()
        #     return sample

        img_name = self.image_list[item]
        mask_name = self.mask_list[item]
        img = Image.open(img_name)
        mask = Image.open(mask_name)
        img = img.convert('RGB')
        mask = mask.convert('L')
        sample = {'image': img, 'mask': mask, 'file_name': self.image_list[item]}

        sample = self.transform(sample)
        sample['file_name'] = self.image_list[item]
        sample['mask'] = (sample['mask']>0.5).float()
        # if not os.path.exists(file_path):
        #     np.save(file_path, sample)   
        
        return sample
    
class SPDatasetDummy(data.Dataset):
    def __init__(self, image_list, mask_list, num_seg, size, compactness, ec, sz):
        self.image_list = image_list
        self.mask_list = mask_list
        
       
        totensor = ToTensorSPDummy(num_seg, compactness, size, ec, sz)
        

        
        self.transform = transforms.Compose([Resize(size), totensor])

        


    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, item):
        file_name = self.image_list[item].split('/')[-1].split('.')[0]+'.npy'
        file_path = os.path.join(str(Path(self.image_list[item]).parents[1]),'VAL', file_name )

        # if self.data_augmentation is False and os.path.exists(file_path):
        #     sample = np.load(file_path, allow_pickle=True).item()
        #     return sample

        img_name = self.image_list[item]
        mask_name = self.mask_list[item]
        img = Image.open(img_name)
        mask = Image.open(mask_name)
        img = img.convert('RGB')
        mask = mask.convert('L')
        sample = {'image': img, 'mask': mask, 'file_name': self.image_list[item]}

        sample = self.transform(sample)
        sample['file_name'] = self.image_list[item]
        sample['mask'] = (sample['mask']>0.5).float()
        # if not os.path.exists(file_path):
        #     np.save(file_path, sample)   
        
        return sample


class DUTSDataset(data.Dataset):
    def __init__(self, image_list, mask_list,  num_seg, size,  data_augmentation=True):
        self.image_list = image_list
        self.mask_list = mask_list
        resolution = int(num_seg**0.5)
        
        if data_augmentation:
            self.transform = transforms.Compose([RandomFlip(0.5),
                          RandomAffine(15, 0.1, 0.1), ResizeDownsample(size),ToTensorRaw(True)])
        else:
            self.transform = transforms.Compose([ResizeDownsample(size), ToTensorRaw(False)])
        # self.centroids = torch.zeros(resolution, resolution, 2).float()
        # for i in range(resolution):
        #     for j in range(resolution):
        #         self.centroids[ i, j, :] = torch.tensor([i, j]).float()
        # self.centroids = self.centroids.reshape(-1, 2)

        


    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, item):
        img_name = self.image_list[item]
        mask_name = self.mask_list[item]
        img = Image.open(img_name)
        mask = Image.open(mask_name)
        img = img.convert('RGB')
        mask = mask.convert('L')
        sample = {'image': img, 'mask': mask, 'file_name': img_name}

        sample = self.transform(sample)
        # sample['features'] = torch.cat((self.centroids, sample['features']), dim=1)
        return sample

class SPDataModule(pl.LightningDataModule):

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
        self.ec = kwargs.get('ec')
        self.aug_strat = kwargs.get('aug_strat')

        
        if not self.skip_train:
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
        data_train = SPDataset(self.tr_image_list, self.tr_mask_list, self.num_seg, self.res, self.compactness, True, self.dataloader, self.coeff, self.ignore_phase)
        return DataLoader(
                data_train, batch_size=self.batch_size, 
                num_workers=self.num_workers, shuffle=True, pin_memory=True, drop_last=True)

    def val_dataloader(self):
        data_val = SPDataset(self.val_image_list, self.val_mask_list,self.num_seg, self.res, self.compactness, False, self.dataloader, self.coeff, self.ignore_phase)
        data_test = SPDataset(self.test_image_list, self.test_mask_list,  self.num_seg, self.res,  self.compactness, False, self.dataloader, self.coeff, self.ignore_phase)
        val_dataloader = DataLoader(
                data_val, batch_size=self.batch_size, 
                num_workers=self.num_workers, pin_memory=True)
        test_dataloader = DataLoader(
                data_test, batch_size=self.batch_size, 
                num_workers=self.num_workers, pin_memory=True)
        return [val_dataloader, test_dataloader]

    def test_dataloader(self):
        data_test = SPDataset(self.test_image_list, self.test_mask_list,  self.num_seg, self.res,  self.compactness, False, self.dataloader, self.coeff, self.ignore_phase)
        return DataLoader(
                data_test, batch_size=self.batch_size, 
                num_workers=self.num_workers, pin_memory=True)




class DUTSDataModule(pl.LightningDataModule):

    def __init__(self, **kwargs):
        super().__init__()

        self.train_dir = kwargs.get('dataset_tr')
        self.test_dir = kwargs.get('dataset_test')
        self.batch_size = kwargs.get('batch_size')
        self.num_seg = kwargs.get('num_seg')
        self.num_workers = kwargs.get('num_workers', 0)
        self.image_size = kwargs.get('size')
        self.debug = kwargs.get('debug')
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

        self.test_image_list = sorted([os.path.join(os.path.join(self.test_dir, 'Image'), f) for f in os.listdir(os.path.join(self.test_dir, 'Image'))])
        self.test_mask_list = sorted([os.path.join(os.path.join(self.test_dir, 'Mask'), f) for f in os.listdir(os.path.join(self.test_dir, 'Mask'))])

        if self.debug:
            if not self.skip_train:
                self.val_image_list = self.val_image_list[:100]
                self.val_mask_list = self.val_mask_list[:100]

                self.tr_image_list = self.tr_image_list[:100]
                self.tr_mask_list = self.tr_mask_list[:100]

            self.test_image_list = self.test_image_list[:100]
            self.test_mask_list = self.test_mask_list[:100]

        
    def train_dataloader(self):
        data_train = DUTSDataset(self.tr_image_list, self.tr_mask_list, self.num_seg, self.image_size,  True)
        return DataLoader(
                data_train, batch_size=self.batch_size, 
                num_workers=self.num_workers, shuffle=True, pin_memory=True, drop_last=True)

    def val_dataloader(self):
        data_val = DUTSDataset(self.val_image_list, self.val_mask_list, self.num_seg, self.image_size, False)
        data_test = DUTSDataset(self.test_image_list, self.test_mask_list, self.num_seg, self.image_size,  False)
        return [DataLoader(
                data_val, batch_size=self.batch_size, 
                num_workers=self.num_workers, pin_memory=True), DataLoader(
                data_test, batch_size=self.batch_size, 
                num_workers=self.num_workers, pin_memory=True)]

    def test_dataloader(self):
        data_test = DUTSDataset(self.test_image_list, self.test_mask_list, self.num_seg, self.image_size,  False)
        return DataLoader(
                data_test, batch_size=self.batch_size, 
                num_workers=self.num_workers, pin_memory=True) 





class SPRSDataModule(pl.LightningDataModule):

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

        self.image_list = np.array(sorted([os.path.join('{}/Image'.format(self.train_dir), f) for f in os.listdir('{}/Image'.format(self.train_dir))]))
        self.mask_list = np.array(sorted([os.path.join('{}/Mask'.format(self.train_dir), f) for f in os.listdir('{}/Mask'.format(self.train_dir))]))

    
        self.tr_image_list = self.image_list
        self.tr_mask_list = self.mask_list

        self.test_image_list = sorted([os.path.join('{}/Image'.format(self.test_dir), f) for f in os.listdir('{}/Image'.format(self.test_dir))])
        self.test_mask_list = sorted([os.path.join('{}/Mask'.format(self.test_dir), f) for f in os.listdir('{}/Mask'.format(self.test_dir))])
        

        
    def train_dataloader(self):
        data_train = SPDataset(self.tr_image_list, self.tr_mask_list, self.num_seg, self.res, self.compactness, True, self.dataloader, self.coeff, self.ignore_phase)
        return DataLoader(
                data_train, batch_size=self.batch_size, 
                num_workers=self.num_workers, shuffle=True, pin_memory=True, drop_last=True)

    def val_dataloader(self):
        data_test = SPDataset(self.test_image_list, self.test_mask_list,  self.num_seg, self.res,  self.compactness, False, self.dataloader, self.coeff, self.ignore_phase)
        val_dataloader = DataLoader(
                data_test, batch_size=self.batch_size, 
                num_workers=self.num_workers, pin_memory=True)
        test_dataloader = DataLoader(
                data_test, batch_size=self.batch_size, 
                num_workers=self.num_workers, pin_memory=True)
        return [val_dataloader, test_dataloader]

    def test_dataloader(self):
        data_test = SPDataset(self.test_image_list, self.test_mask_list,  self.num_seg, self.res,  self.compactness, False, self.dataloader, self.coeff, self.ignore_phase)
        return DataLoader(
                data_test, batch_size=self.batch_size, 
                num_workers=self.num_workers, pin_memory=True)