import os
import torch.utils.data as data
import torchvision.transforms as transforms
from collections import defaultdict
import numpy as np
import torch
from PIL import Image
from skimage.segmentation import slic
from skimage.measure import regionprops, regionprops_table

class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        img, mask = sample['image'], sample['mask']
        img, mask = img.resize((self.size, self.size), resample=Image.BILINEAR), mask.resize((self.size, self.size),
                                                                                             resample=Image.BILINEAR)
        return {'image': img, 'mask': mask}


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        img, mask = sample['image'], sample['mask']
        img, mask = img.resize((350, 350), resample=Image.BILINEAR), mask.resize((350, 350), resample=Image.BILINEAR)
        h, w = img.size
        new_h, new_w = self.size, self.size

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


class ToTensor(object):
    def __init__(self, seq_len):
        self.tensor = transforms.ToTensor()
        self.seq_len = seq_len

    def __call__(self, sample):
        img, mask = sample['image'], sample['mask']
        img_np = np.array(img)
        img_size = img_np.shape[1]
        mask_np = np.array(mask)/255.
        num_seg = 600
        segments = slic(img_np, n_segments=num_seg,
            compactness=10.0,
            max_num_iter=10,
            convert2lab=True,
            enforce_connectivity=False,
            slic_zero=True,
            min_size_factor=0.,)

        vs_right = np.vstack([segments[:,:-1].ravel(), segments[:,1:].ravel()])
        vs_below = np.vstack([segments[:-1,:].ravel(), segments[1:,:].ravel()])
        bneighbors = np.unique(np.hstack([vs_right, vs_below]), axis=1)
        dict = defaultdict(lambda: [0]*9)
        for ref, nb in zip(bneighbors[0], bneighbors[1]):
            if dict[ref][0]==0:
                dict[ref].pop(0)
                dict[ref].append(nb)

        regions = regionprops_table(segments, intensity_image=img_np, properties=('label', 'centroid', 'area', 'intensity_mean', 'extent', 'coords', 'eccentricity'))
        seq_len = self.seq_len
        features = np.zeros([seq_len, 8])
        seq_mask = np.zeros([seq_len])
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

        neighbor_array = np.zeros([seq_len, 9])
        for key, value in dict.items():
            neighbor_array[key-1] = value

        features, neighbor_array, seq_mask, segments, mask, img = torch.tensor(features).float(), torch.tensor(neighbor_array).long(), torch.tensor(seq_mask).float(), torch.tensor(segments), self.tensor(mask), self.tensor(img)
        return {'features': features, 'seq_mask': seq_mask, 'segments': segments, 'mask': mask, 'img': img, 'neighbor_array': neighbor_array}


class DUTSDataset(data.Dataset):
    def __init__(self, root_dir, seq_len, train=True, data_augmentation=True):
        self.root_dir = root_dir
        self.train = train
        self.image_list = sorted(os.listdir('{}/DUTS-{}-Image'.format(root_dir, 'TR' if train else 'TE')))
        self.mask_list = sorted(os.listdir('{}/DUTS-{}-Mask'.format(root_dir, 'TR' if train else 'TE')))
        self.transform = transforms.Compose(
            [RandomFlip(0.5),
             RandomCrop(300),
             ToTensor(seq_len)])
        if not (train and data_augmentation):
            self.transform = transforms.Compose([Resize(300), ToTensor(seq_len)])
        self.root_dir = root_dir
        self.train = train
        self.data_augmentation = data_augmentation

    def arrange(self):
        flag = True
        if len(self.image_list) > len(self.mask_list):
            for image in self.image_list:
                for mask in self.mask_list:
                    if image.split("Image")[-1].split(".")[-2] == mask.split("Mask")[-1].split(".")[-2]:
                        print(image.split("Image")[-1].split(".")[-2])
                        flag = False
                if flag:
                    print(image + ' Deleted')
                    os.remove('{}/DUTS-{}-Image/{}'.format(self.root_dir, 'TR' if self.train else 'TE', image))
        else:
            for mask in self.mask_list:
                for image in self.image_list:
                    if image.split("Image")[-1].split(".")[-2] == mask.split("Mask")[-1].split(".")[-2]:
                        print(image.split("Image")[-1].split(".")[-2])
                        flag = False
                if flag:
                    print(mask + ' Deleted')
                    os.remove('{}/DUTS-{}-Mask/{}'.format(self.root_dir, 'TR' if self.train else 'TE', mask))
        self.image_list = sorted(os.listdir('{}/DUTS-{}-Image'.format(self.root_dir, 'TR' if self.train else 'TE')))
        self.mask_list = sorted(os.listdir('{}/DUTS-{}-Mask'.format(self.root_dir, 'TR' if self.train else 'TE')))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, item):
        img_name = '{}/DUTS-{}-Image/{}'.format(self.root_dir, 'TR' if self.train else 'TE', self.image_list[item])
        mask_name = '{}/DUTS-{}-Mask/{}'.format(self.root_dir, 'TR' if self.train else 'TE', self.mask_list[item])
        img = Image.open(img_name)
        mask = Image.open(mask_name)
        img = img.convert('RGB')
        mask = mask.convert('L')
        sample = {'image': img, 'mask': mask}

        sample = self.transform(sample)
        return sample


class PairDataset(data.Dataset):
    def __init__(self, root_dir, train=True, data_augmentation=True):
        self.root_dir = root_dir
        self.train = train
        self.image_list = sorted(os.listdir(os.path.join(root_dir, 'images')))
        self.mask_list = sorted(os.listdir(os.path.join(root_dir, 'masks')))
        self.transform = transforms.Compose(
            [RandomFlip(0.5),
             RandomCrop(224),
             ToTensor()])
        if not (train and data_augmentation):
            self.transform = transforms.Compose([Resize(224), ToTensor()])
        self.root_dir = root_dir
        self.data_augmentation = data_augmentation

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, item):
        img_name = os.path.join(self.root_dir, 'images', self.image_list[item])
        mask_name = os.path.join(self.root_dir, 'masks', self.mask_list[item])
        img = Image.open(img_name)
        mask = Image.open(mask_name)
        img = img.convert('RGB')
        mask = mask.convert('L')
        sample = {'image': img, 'mask': mask}

        sample = self.transform(sample)
        return sample


class CustomDataset(data.Dataset):
    def __init__(self, root_dir):
        self.image_list = sorted(os.listdir(root_dir))
        self.transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
        self.root_dir = root_dir

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, item):
        img_name = '{}/{}'.format(self.root_dir, self.image_list[item])
        img = Image.open(img_name)
        sample = img.convert('RGB')
        sample = self.transform(sample)
        return sample


if __name__ == '__main__':
    ds = DUTSDataset('../DUTS-TR')
    ds.arrange()
