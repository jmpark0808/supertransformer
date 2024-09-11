#!/usr/bin/env python
#
# Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>
#

"""Download the Swedish Traffic Signs dataset and create the Speed Limit Signs
dataset from and train with attention sampling.

NOTE: Swedish Traffic Signs dataset is provided from
      https://www.cvl.isy.liu.se/research/datasets/traffic-signs-dataset/ .
"""

import argparse
from collections import namedtuple
from functools import partial
from PIL import Image
import hashlib
import urllib.request
import os
from os import path
import string
import sys
import zipfile
import torch.utils.data as data
from cv2 import imread, imwrite
import cv2
import torch
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from skimage.segmentation import slic
from dataset.attributes import *
from skimage.measure import regionprops_table
from skimage.feature import local_binary_pattern
from skimage import segmentation, color
from dataset.fft_transform import *
from dataset.randaugment import RandAugment
import pytorch_lightning as pl
import matplotlib.patches as patches



def check_file(filepath, md5sum):
    """Check a file against an md5 hash value.

    Returns
    -------
        True if the file exists and has the given md5 sum False otherwise
    """
    try:
        md5 = hashlib.md5()
        with open(filepath, "rb") as f:
            for chunk in iter(partial(f.read, 4096), b""):
                md5.update(chunk)
        return md5.hexdigest() == md5sum
    except FileNotFoundError:
        return False


def download_file(url, destination, progress_file=sys.stderr):
    """Download a file with progress."""
    response = urllib.request.urlopen(url)
    n_bytes = response.headers.get("Content-Length")
    if n_bytes == "":
        n_bytes = 0
    else:
        n_bytes = int(n_bytes)

    message = "\rReceived {} / {}"
    cnt = 0
    with open(destination, "wb") as dst:
        while True:
            print(message.format(cnt, n_bytes), file=progress_file,
                  end="", flush=True)
            data = response.read(65535)
            if len(data) == 0:
                break
            dst.write(data)
            cnt += len(data)
    print(file=progress_file)


def ensure_dataset_exists(directory, tries=1, progress_file=sys.stderr):
    """Ensure that the dataset is downloaded and is correct.

    Correctness is checked only against the annotations files.
    """
    set1_url = ("http://www.isy.liu.se/cvl/research/trafficSigns"
                "/swedishSignsSummer/Set1/Set1Part0.zip")
    set1_annotations_url = ("http://www.isy.liu.se/cvl/research/trafficSigns"
                            "/swedishSignsSummer/Set1/annotations.txt")
    set1_annotations_md5 = "9106a905a86209c95dc9b51d12f520d6"
    set2_url = ("http://www.isy.liu.se/cvl/research/trafficSigns"
                "/swedishSignsSummer/Set2/Set2Part0.zip")
    set2_annotations_url = ("http://www.isy.liu.se/cvl/research/trafficSigns"
                            "/swedishSignsSummer/Set2/annotations.txt")
    set2_annotations_md5 = "09debbc67f6cd89c1e2a2688ad1d03ca"

    integrity = (
        check_file(
            path.join(directory, "Set1", "annotations.txt"),
            set1_annotations_md5
        ) and check_file(
            path.join(directory, "Set2", "annotations.txt"),
            set2_annotations_md5
        )
    )

    if integrity:
        return

    if tries <= 0:
        raise RuntimeError(("Cannot download dataset or dataset download "
                            "is corrupted"))

    print("Downloading Set1", file=progress_file)
    download_file(set1_url, path.join(directory, "Set1.zip"),
                  progress_file=progress_file)
    print("Extracting...", file=progress_file)
    with zipfile.ZipFile(path.join(directory, "Set1.zip")) as archive:
        archive.extractall(path.join(directory, "Set1"))
    print("Getting annotation file", file=progress_file)
    download_file(
        set1_annotations_url,
        path.join(directory, "Set1", "annotations.txt"),
        progress_file=progress_file
    )
    print("Downloading Set2", file=progress_file)
    download_file(set2_url, path.join(directory, "Set2.zip"),
                  progress_file=progress_file)
    print("Extracting...", file=progress_file)
    with zipfile.ZipFile(path.join(directory, "Set2.zip")) as archive:
        archive.extractall(path.join(directory, "Set2"))
    print("Getting annotation file", file=progress_file)
    download_file(
        set2_annotations_url,
        path.join(directory, "Set2", "annotations.txt"),
        progress_file=progress_file
    )

    return ensure_dataset_exists(
        directory,
        tries=tries-1,
        progress_file=progress_file
    )


class Sign(namedtuple("Sign", ["visibility", "bbox", "type", "name"])):
    """A sign object. Useful for making ground truth images as well as making
    the dataset."""
    @property
    def x_min(self):
        return self.bbox[2]

    @property
    def x_max(self):
        return self.bbox[0]

    @property
    def y_min(self):
        return self.bbox[3]

    @property
    def y_max(self):
        return self.bbox[1]

    @property
    def area(self):
        return (self.x_max - self.x_min) * (self.y_max - self.y_min)

    @property
    def center(self):
        return [
            (self.y_max - self.y_min)/2 + self.y_min,
            (self.x_max - self.x_min)/2 + self.x_min
        ]

    @property
    def visibility_index(self):
        visibilities = ["VISIBLE", "BLURRED", "SIDE_ROAD", "OCCLUDED"]
        return visibilities.index(self.visibility)

    def pixels(self, scale, size):
        return zip(*(
            (i, j)
            for i in range(round(self.y_min*scale), round(self.y_max*scale)+1)
            for j in range(round(self.x_min*scale), round(self.x_max*scale)+1)
            if i < round(size[0]*scale) and j < round(size[1]*scale)
        ))

    def __lt__(self, other):
        if not isinstance(other, Sign):
            raise ValueError("Signs can only be compared to signs")

        if self.visibility_index != other.visibility_index:
            return self.visibility_index < other.visibility_index

        return self.area > other.area


class STS:
    """The STS class reads the annotations and creates the corresponding
    Sign objects."""
    def __init__(self, directory, train=True, seed=0):
        ensure_dataset_exists(directory)

        self._directory = directory
        self._inner = "Set{}".format(1 + ((seed + 1 + int(train)) % 2))
        self._data = self._load_signs(self._directory, self._inner)

    def _load_files(self, directory, inner):
        files = set()
        with open(path.join(directory, inner, "annotations.txt")) as f:
            for l in f:
                files.add(l.split(":", 1)[0])
        return sorted(files)

    def _read_bbox(self, parts):
        def _float(x):
            try:
                return float(x)
            except ValueError:
                if len(x) > 0:
                    return _float(x[:-1])
                raise
        return [_float(x) for x in parts]

    def _load_signs(self, directory, inner):
        with open(path.join(directory, inner, "annotations.txt")) as f:
            lines = [l.strip() for l in f]
        keys, values = zip(*(l.split(":", 1) for l in lines))
        all_signs = []
        for v in values:
            signs = []
            for sign in v.split(";"):
                if sign == [""] or sign == "":
                    continue
                parts = [s.strip() for s in sign.split(",")]
                if parts[0] == "MISC_SIGNS":
                    continue
                signs.append(Sign(
                    visibility=parts[0],
                    bbox=self._read_bbox(parts[1:5]),
                    type=parts[5],
                    name=parts[6]
                ))
            all_signs.append(signs)
        images = [path.join(directory, inner, f) for f in keys]

        return list(zip(images, all_signs))

    def __len__(self):
        return len(self._data)

    def __getitem__(self, i):
        return self._data[i]




class SpeedLimits(data.Dataset):
    """Provide a Keras Sequence for the SpeedLimits dataset which is basically
    a filtered version of the STS dataset.

    Arguments
    ---------
        directory: str, The directory that the dataset already is or is going
                   to be downloaded in
        train: bool, Select the training or testing sets
        seed: int, The prng seed for the dataset
    """
    LIMITS = ["50_SIGN", "70_SIGN", "80_SIGN"]
    CLASSES = ["EMPTY", *LIMITS]

    def __init__(self, directory, train=True, seed=0):
        self._data = self._filter(STS(directory, train, seed))

    def _filter(self, data):
        filtered = []
        for image, signs in data:
            signs, acceptable = self._acceptable(signs)
            if acceptable:
                if not signs:
                    filtered.append((image, 0))
                else:
                    filtered.append((image, self.CLASSES.index(signs[0].name)))
        return filtered

    def _acceptable(self, signs):
        # Keep it as empty
        if not signs:
            return signs, True

        # Filter just the speed limits and sort them wrt visibility
        signs = sorted(s for s in signs if s.name in self.LIMITS)

        # No speed limit but many other signs
        if not signs:
            return None, False

        # Not visible sign so skip
        if signs[0].visibility != "VISIBLE":
            return None, False

        return signs, True

    def __len__(self):
        return len(self._data)

    def __getitem__(self, i):
        image, category = self._data[i]

        data = imread(image)
        data_l = cv2.resize(data, (320, 240))
        
        data_l = data_l.astype(np.float32)/np.float32(255.)
        data_h = data.astype(np.float32) / np.float32(255.)
        # label = np.eye(len(self.CLASSES), dtype=np.float32)[category]



        return torch.tensor(data_l).permute(2, 0, 1), torch.tensor(data_h).permute(2, 0, 1), torch.tensor(category)

    @property
    def image_size(self):
        return self[0][0].shape[:2]

    @property
    def class_frequencies(self):
        """Compute and return the class specific frequencies."""
        freqs = np.zeros(len(self.CLASSES), dtype=np.float32)
        for image, category in self._data:
            freqs[category] += 1
        return freqs/len(self._data)

    def strided(self, N):
        """Extract N images almost in equal proportions from each category."""
        order = np.arange(len(self._data))
        np.random.shuffle(order)
        idxs = []
        cat = 0
        while len(idxs) < N:
            for i in order:
                image, category = self._data[i]
                if cat == category:
                    idxs.append(i)
                    cat = (cat + 1) % len(self.CLASSES)
                if len(idxs) >= N:
                    break
        return idxs
    

class SpeedLimitsCropDataset(data.Dataset):
    def __init__(self, root_dir, augmentation, coeff):
        self.root_dir = root_dir
        self.image_list = []
        self.target_list = []
        self.coeff = coeff
        self.augmentation = augmentation
        for file in os.listdir(root_dir):
            if '_target' in file:
                self.image_list.append(os.path.join(root_dir, file.split('_target')[0]+'.npy'))
                self.target_list.append(os.path.join(root_dir, file)) 
            else:
               continue
             
            
                

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, item):
        
        features_np = np.load(self.image_list[item])

        if self.augmentation:
            features_np = horizontal_flip(features_np, self.coeff, 0.5, 1280,(192, 256))
            features_np = rotate(features_np, self.coeff, 15, 0.5, (960//2, 1280//2))

            


        features = torch.tensor(features_np).float()
        if self.augmentation:

            randaug = RandAugment(5)
            color_space = features[:, 3:6].reshape(192, 256, 3).permute(2, 0, 1)
            color_space = (color_space*255).to(torch.uint8)
            color_space = randaug(color_space).float()
            color_space /= 255.
            # plt.imshow(color_space.permute(1, 2, 0).detach().cpu().numpy())
            # plt.show()
            color_space = color_space.reshape(3, 192*256).permute(1, 0)
            
            features[:, 3:6] = color_space

        target = torch.squeeze(torch.tensor(np.load(self.target_list[item])))

        return features, target


class SpeedLimitsCropExport(data.Dataset):
    """Provide a Keras Sequence for the SpeedLimits dataset which is basically
    a filtered version of the STS dataset.

    Arguments
    ---------
        directory: str, The directory that the dataset already is or is going
                   to be downloaded in
        train: bool, Select the training or testing sets
        seed: int, The prng seed for the dataset
    """
    LIMITS = ["50_SIGN", "70_SIGN", "80_SIGN"]
    CLASSES = ["EMPTY", *LIMITS]

    def __init__(self, directory, compactness, num_seg, coeff, train=True, seed=0, export_dir=None):
        self._data = self._filter(STS(directory, train, seed))
        self.export_dir = export_dir
        assert export_dir is not None, 'Export Dir must not be none'
        os.makedirs(self.export_dir, exist_ok=True)
        self.compactness = compactness
        self.num_seg = num_seg
        self.coeff = coeff
        

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

    def _filter(self, data):
        filtered = []
        for image, signs in data:
            signs, acceptable = self._acceptable(signs)
            if acceptable:
                if not signs:
                    filtered.append((image, 0, None))
                else:
                    filtered.append((image, self.CLASSES.index(signs[0].name), signs[0].bbox))
        return filtered

    def _acceptable(self, signs):
        # Keep it as empty
        if not signs:
            return signs, True

        # Filter just the speed limits and sort them wrt visibility
        signs = sorted(s for s in signs if s.name in self.LIMITS)

        # No speed limit but many other signs
        if not signs:
            return None, False

        # Not visible sign so skip
        if signs[0].visibility != "VISIBLE":
            return None, False

        return signs, True

    def __len__(self):
        return len(self._data)

    def __getitem__(self, i):
        image, category, bbox = self._data[i]

        sp_file_name = image.split('/')[-1].split('.')[0]+'.npy'
        sp_file_target = image.split('/')[-1].split('.')[0]+'_target.npy'
      

        sp_file_path = os.path.join(self.export_dir, sp_file_name)
        sp_file_path_target = os.path.join(self.export_dir, sp_file_target)
     

        data = imread(image) # 960 x 1280
        img_gray = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
        ### Included
        data = np.array(data[...,::-1])
        data_gray = np.array(img_gray)
        if bbox is not None:

            data = data[int(bbox[3]):int(bbox[1]), int(bbox[2]):int(bbox[0])]
            data_gray = data_gray[int(bbox[3]):int(bbox[1]), int(bbox[2]):int(bbox[0])]
        else:
            random_size = np.random.randint(100, 200)
            random_height_start = np.random.randint(0, 960-random_size)
            random_width_start = np.random.randint(0, 1280-random_size)
            data = data[random_height_start:random_height_start+random_size, random_width_start:random_width_start+random_size]
            data_gray = data_gray[random_height_start:random_height_start+random_size, random_width_start:random_width_start+random_size]


        data = torch.tensor(data).permute(2, 0, 1) # 3, H, W
        data_gray = torch.tensor(data_gray)
        
        pad = [0, 0, 0, 0]
        img_size = 224
        if data.size(1) %2 == 0:
            half = (img_size-data.size(1))//2
            pad[2] = half
            pad[3] = half
        else:
            half_l = (img_size-data.size(1))//2
            half_r = (img_size-data.size(1))//2+1
            pad[2] = half_l
            pad[3] = half_r

        if data.size(2) %2 == 0:
            half = (img_size-data.size(2))//2
            pad[0] = half
            pad[1] = half
        else:
            half_l = (img_size-data.size(2))//2
            half_r = (img_size-data.size(2))//2+1
            pad[0] = half_l
            pad[1] = half_r

        

        data = torch.nn.functional.pad(data, tuple(pad), 'constant', 0)
        data_gray = torch.nn.functional.pad(data_gray, tuple(pad), 'constant', 0)

        

        img_np = data.permute(1, 2, 0).detach().numpy()
        img_gray_np = data_gray.detach().numpy()

        

        segments = slic(img_np, n_segments=self.num_seg,
            compactness=self.compactness,
            max_num_iter=10,
            convert2lab=True,
            enforce_connectivity=False,
            slic_zero=False)
        
        # out = color.label2rgb(segments, img_np, kind='avg', bg_label=0)
        # out = segmentation.mark_boundaries(out, segments, (0, 0, 0))
        # plt.imshow(out)
        # plt.show()
        lbp_np = local_binary_pattern(img_gray_np, 8, 1, method='uniform')
        regions_lbp = regionprops_table(segments, intensity_image=lbp_np, extra_properties=[self.lbp])
        regions = regionprops_table(segments, intensity_image=img_np, properties=('label', 'centroid', 'intensity_mean',
                                                                                    'coords'), extra_properties=[image_stdev, self.fourier_descriptors])#, polarize])

        label = regions['label']
  
        features = np.zeros([self.num_seg, 8+(self.coeff)*2+10])
    
        # if self.ignore_phase:
        #     features = np.zeros([self.num_seg, 8+self.coeff])
        #     for i in range(self.coeff):
        #         features[label-1, 8+i] = regions[f'fourier_descriptors-{i}']
        # else:
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
        
        np.save(sp_file_path, features)
        np.save(sp_file_path_target, np.array([category]))
        
        return torch.empty(0)
            


        ##### end include
        # data_l = cv2.resize(data, (320, 240))
        
        # data_l = data_l.astype(np.float32)/np.float32(255.)
        # data_h = data.astype(np.float32) / np.float32(255.)
        # # label = np.eye(len(self.CLASSES), dtype=np.float32)[category]



        # return torch.tensor(data_l).permute(2, 0, 1), torch.tensor(data_h).permute(2, 0, 1), torch.tensor(category)

    @property
    def image_size(self):
        return self[0][0].shape[:2]

    @property
    def class_frequencies(self):
        """Compute and return the class specific frequencies."""
        freqs = np.zeros(len(self.CLASSES), dtype=np.float32)
        for image, category in self._data:
            freqs[category] += 1
        return freqs/len(self._data)

    def strided(self, N):
        """Extract N images almost in equal proportions from each category."""
        order = np.arange(len(self._data))
        np.random.shuffle(order)
        idxs = []
        cat = 0
        while len(idxs) < N:
            for i in order:
                image, category = self._data[i]
                if cat == category:
                    idxs.append(i)
                    cat = (cat + 1) % len(self.CLASSES)
                if len(idxs) >= N:
                    break
        return idxs
    
class SpeedLimitsDataset(data.Dataset):
    def __init__(self, root_dir, augmentation, coeff):
        self.root_dir = root_dir
        self.image_list = []
        self.target_list = []
        self.coeff = coeff
        self.augmentation = augmentation
        for file in os.listdir(root_dir):
            if '_target' in file:
                self.image_list.append(os.path.join(root_dir, file.split('_target')[0]+'.npy'))
                self.target_list.append(os.path.join(root_dir, file)) 
            else:
               continue
             
            
                

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, item):
        
        features_np = np.load(self.image_list[item])

        if self.augmentation:
            features_np = horizontal_flip(features_np, self.coeff, 0.5, 1280, (56, 56))
            features_np = rotate(features_np, self.coeff, 15, 0.5, (112, 112))

            


        features = torch.tensor(features_np).float()
        if self.augmentation:

            randaug = RandAugment(5)
            color_space = features[:, 3:6].reshape(56, 56, 3).permute(2, 0, 1)
            color_space = (color_space*255).to(torch.uint8)
            color_space = randaug(color_space).float()
            color_space /= 255.
            # plt.imshow(color_space.permute(1, 2, 0).detach().cpu().numpy())
            # plt.show()
            color_space = color_space.reshape(3, 56*56).permute(1, 0)
            
            features[:, 3:6] = color_space

        target = torch.squeeze(torch.tensor(np.load(self.target_list[item])))

        return features, target


class SpeedLimitsExport(data.Dataset):
    """Provide a Keras Sequence for the SpeedLimits dataset which is basically
    a filtered version of the STS dataset.

    Arguments
    ---------
        directory: str, The directory that the dataset already is or is going
                   to be downloaded in
        train: bool, Select the training or testing sets
        seed: int, The prng seed for the dataset
    """
    LIMITS = ["50_SIGN", "70_SIGN", "80_SIGN"]
    CLASSES = ["EMPTY", *LIMITS]

    def __init__(self, directory, compactness, num_seg, coeff, train=True, seed=0, export_dir=None):
        self._data = self._filter(STS(directory, train, seed))
        self.export_dir = export_dir
        assert export_dir is not None, 'Export Dir must not be none'
        os.makedirs(self.export_dir, exist_ok=True)
        self.compactness = compactness
        self.num_seg = num_seg
        self.coeff = coeff
        

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

    def _filter(self, data):
        filtered = []
        for image, signs in data:
            signs, acceptable = self._acceptable(signs)
            if acceptable:
                if not signs:
                    filtered.append((image, 0))
                else:
                    filtered.append((image, self.CLASSES.index(signs[0].name)))
        return filtered

    def _acceptable(self, signs):
        # Keep it as empty
        if not signs:
            return signs, True

        # Filter just the speed limits and sort them wrt visibility
        signs = sorted(s for s in signs if s.name in self.LIMITS)

        # No speed limit but many other signs
        if not signs:
            return None, False

        # Not visible sign so skip
        if signs[0].visibility != "VISIBLE":
            return None, False

        return signs, True

    def __len__(self):
        return len(self._data)

    def __getitem__(self, i):
        image, target = self._data[i]

        sp_file_name = image.split('/')[-1].split('.')[0]+'.npy'
        sp_file_target = image.split('/')[-1].split('.')[0]+'_target.npy'
      

        sp_file_path = os.path.join(self.export_dir, sp_file_name)
        sp_file_path_target = os.path.join(self.export_dir, sp_file_target)
     

        img = Image.open(image)
        img_gray = img.convert('L')
        img = img.convert('RGB')
    
        img_np = np.ascontiguousarray(img).astype(np.uint8)
        img_gray_np = np.ascontiguousarray(img_gray).astype(np.uint8)
        
        segments = slic(img_np, n_segments=self.num_seg,
            compactness=self.compactness,
            max_num_iter=10,
            convert2lab=True,
            enforce_connectivity=False,
            slic_zero=False)
        
        # out = color.label2rgb(segments, img_np, kind='avg', bg_label=0)
        # out = segmentation.mark_boundaries(out, segments, (0, 0, 0))
        # plt.imshow(out)
        # plt.show()
        lbp_np = local_binary_pattern(img_gray_np, 8, 1, method='uniform')
        regions_lbp = regionprops_table(segments, intensity_image=lbp_np, extra_properties=[self.lbp])
        regions = regionprops_table(segments, intensity_image=img_np, properties=('label', 'centroid', 'intensity_mean',
                                                                                    'coords'), extra_properties=[image_stdev, self.fourier_descriptors])#, polarize])

        label = regions['label']
        features = np.zeros([self.num_seg, 8+(self.coeff)*2+10])
    
        # if self.ignore_phase:
        #     features = np.zeros([self.num_seg, 8+self.coeff])
        #     for i in range(self.coeff):
        #         features[label-1, 8+i] = regions[f'fourier_descriptors-{i}']
        # else:
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
        
        np.save(sp_file_path, features)
        np.save(sp_file_path_target, np.array([target]))
        
        return torch.empty(0)

    @property
    def image_size(self):
        return self[0][0].shape[:2]

    @property
    def class_frequencies(self):
        """Compute and return the class specific frequencies."""
        freqs = np.zeros(len(self.CLASSES), dtype=np.float32)
        for image, category in self._data:
            freqs[category] += 1
        return freqs/len(self._data)

    def strided(self, N):
        """Extract N images almost in equal proportions from each category."""
        order = np.arange(len(self._data))
        np.random.shuffle(order)
        idxs = []
        cat = 0
        while len(idxs) < N:
            for i in order:
                image, category = self._data[i]
                if cat == category:
                    idxs.append(i)
                    cat = (cat + 1) % len(self.CLASSES)
                if len(idxs) >= N:
                    break
        return idxs

class SPSpeedLimitsDataModule(pl.LightningDataModule):

    def __init__(self, **kwargs):
        super().__init__()

        
        train_dir = kwargs.get('train_dir')
        test_dir = kwargs.get('test_dir')
        self.batch_size = kwargs.get('batch_size')
        self.num_workers = kwargs.get('num_workers', 0)
        self.num_seg = kwargs.get('num_seg', 600)
        self.coeff = kwargs.get('coeff', 70)
        self.compactness = kwargs.get('compactness', 10)
        self.seed = kwargs.get('seed')
        self.dilation = kwargs.get('dilation')
        self.size = kwargs.get('size')
    

        train_dataset = SpeedLimitsDataset(train_dir, True, self.coeff)
        test_dataset = SpeedLimitsDataset(test_dir, False, self.coeff)

        self.train_source_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True,
                                                               num_workers =self.num_workers, drop_last=True)
        
        self.test_source_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False,
                                                              num_workers=self.num_workers, drop_last=False)

        
    def train_dataloader(self):
        return self.train_source_loader

    def val_dataloader(self):
        return self.test_source_loader

    def test_dataloader(self):
        return self.test_source_loader


class SpeedLimitsDataModule(pl.LightningDataModule):

    def __init__(self, **kwargs):
        super().__init__()

        
        train_dir = kwargs.get('train_dir')
        test_dir = kwargs.get('test_dir')
        self.batch_size = kwargs.get('batch_size')
        self.num_workers = kwargs.get('num_workers', 0)
        self.num_seg = kwargs.get('num_seg', 600)
        self.coeff = kwargs.get('coeff', 70)
        self.compactness = kwargs.get('compactness', 10)
        self.seed = kwargs.get('seed')
        self.dilation = kwargs.get('dilation')
        self.size = kwargs.get('size')
    

        train_dataset = SpeedLimits(train_dir, True)
        test_dataset = SpeedLimits(test_dir, False)

        self.train_source_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True,
                                                               num_workers =self.num_workers, drop_last=True)
        
        self.test_source_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False,
                                                              num_workers=self.num_workers, drop_last=False)

        
    def train_dataloader(self):
        return self.train_source_loader

    def val_dataloader(self):
        return self.test_source_loader

    def test_dataloader(self):
        return self.test_source_loader



def main(argv):
    parser = argparse.ArgumentParser(
        description=("Fetch the Sweidish Traffic Signs dataset and parse "
                     "it into the Speed Limits dataset subset")
    )
    parser.add_argument(
        "dataset",
        help="The location to download the dataset to"
    )
    

    args = parser.parse_args(argv)

    # Load the data
    training_set = SpeedLimitsCropExport(args.dataset, 10, 3136, 10, train=True, export_dir='/mnt/dragon/Datasets/stopsigns/Toy_TR')
    test_set = SpeedLimitsCropExport(args.dataset, 10, 3136, 10, train=False, export_dir='/mnt/dragon/Datasets/stopsigns/Toy_TE')
    # training_set = SpeedLimits(args.dataset, train=True)
    # test_set = SpeedLimits(args.dataset, train=False)
    training_batched = DataLoader(training_set, 1, num_workers=10)
    test_batched = DataLoader(test_set, 1, num_workers=10)
    # all_dims = []
    # all_areas = []
    # print(len(training_set))
    # print(len(test_set))
    for batch in training_batched:
        pass
        # if crop.size(2) >= 200 or crop.size(3) >= 200:
        #     assert(0)
        # all_dims.append(crop.size())
        # all_areas.append(crop.size(2)*crop.size(3))
        
        # print(torch.min(data), torch.max(data))
        # plt.imshow(data.detach().numpy()[0])
        # plt.show()
        

    for batch in test_batched:
        pass
        # if crop.size(2) >= 200 or crop.size(3) >= 200:
        #     assert(0)
        # all_dims.append(crop.size())
        # all_areas.append(crop.size(2)*crop.size(3))
        # pass
    # print(all_dims)
    # print(all_dims[torch.argmax(torch.tensor(all_areas))])



if __name__ == "__main__":
    main(None)