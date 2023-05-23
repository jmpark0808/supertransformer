import os
import sys
sys.path.insert(0, '/home/abcd/abcde/supertransformer')
import matplotlib.pyplot as plt
import cv2
from skimage import io
from skimage.segmentation import mark_boundaries, slic
from skimage.measure import regionprops_table
import numpy as np
from PIL import Image
from tqdm import tqdm
import pickle
from Analysis.contour_resample import resample_2d


dataset_images = '/mnt/hdd/Datasets/DUTS/DUTS-TR/Image'
masks = '/mnt/hdd/Datasets/DUTS/DUTS-TR/Mask'

cts = []
areas = []

def fft(region):
    # note the ddof arg to get the sample var if you so desire!
    region = (region.astype(int)*255).astype(np.uint8)
    rows, cols = region.shape[-2:]
    contour = fourier_descriptor(region, 9, rows, cols)

    return contour

def contour(region):
    region = (region.astype(int)*255).astype(np.uint8)
    contour_og, hierarchy = cv2.findContours(region, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    return contour_og[0]




def fourier_descriptor(binary_img, degree, rows, cols):

    contour, hierarchy = cv2.findContours(binary_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    contour_array = contour[0][:, 0, :]
    xi, yi = resample_2d(contour_array, 70)
    contour_array = np.stack((xi, yi), axis=1)
    contour_complex = np.empty(contour_array.shape[:-1], dtype=complex)
    contour_complex.real = contour_array[:, 0]
    contour_complex.imag = contour_array[:, 1]
    fourier_result = np.fft.fft(contour_complex)
    # fourier_truncated = truncate_descriptor(fourier_result, degree)
    contour_reconstructed = reconstruct(fourier_result, degree, rows, cols)
    return contour_reconstructed


def reconstruct(descriptors, degree, rows, cols):
    """ reconstruct(descriptors, degree) attempts to reconstruct the image
    using the first [degree] descriptors of descriptors"""
    # truncate the long list of descriptors to certain length

    descriptor_in_use = truncate_descriptor(descriptors, degree)
    contour_reconstruct = np.fft.ifft(descriptor_in_use)
    contour_reconstruct = np.array(
        [contour_reconstruct.real, contour_reconstruct.imag])
    contour_reconstruct = np.transpose(contour_reconstruct)
    contour_reconstruct = np.expand_dims(contour_reconstruct, axis=1)
    # make positive
    # if contour_reconstruct.min() < 0:
    #     contour_reconstruct -= contour_reconstruct.min()
    # # normalization
    # contour_reconstruct /=  contour_reconstruct.max()
    # contour_reconstruct[:, :, 0] *= rows
    # contour_reconstruct[:, :, 1] *= cols
    # type cast to int32
    contour_reconstruct = contour_reconstruct.astype(np.int32, copy=False)
    
    return contour_reconstruct


def truncate_descriptor(descriptors, degree):
    """this function truncates an unshifted fourier descriptor array
    and returns one also unshifted"""

    descriptors = np.fft.fftshift(descriptors)
    center_index = len(descriptors) // 2
    # descriptors = descriptors[center_index - degree // 2:center_index + degree // 2]
    descriptors[:center_index - degree // 2] = 0
    descriptors[center_index + degree // 2:] = 0
    descriptors = np.fft.ifftshift(descriptors)
    # descriptors[degree:] = 0
    return descriptors
    

for file in tqdm(os.listdir(dataset_images)):
    name = file.split('.jpg')[0]
    image = os.path.join(dataset_images, name+'.jpg')
    mask = os.path.join(masks, name+'.png')

    img = Image.open(image)
    msk = Image.open(mask)
    img = img.convert('RGB').resize((300, 300))
    msk = msk.convert('L').resize((300, 300))
    img = np.array(img)
    msk = np.array(msk)
    
    # msk[msk>125] = 255
    # msk[msk<=125] = 0

    # empty_background = np.zeros_like(msk)

    # msk_boundaries = np.sum(mark_boundaries(empty_background, msk), axis=2)

    msk[msk<=125] = 0
    msk[msk>125] = 1
    

    segments_ec = slic(img, n_segments=625,
    compactness=10,
    max_num_iter=10,
    convert2lab=True,
    enforce_connectivity=True,
    slic_zero=False)

    regions = regionprops_table(segments_ec, intensity_image=img, properties=('label', 'centroid', 'area', 'bbox', 'image'), extra_properties=[fft, contour])
    fig, ax = plt.subplots(1, 2)
    print(regions.keys())
    # np.save('/home/abcd/abcde/supertransformer/Analysis/sample_sp',regions['image'][500])

    for contours, y, x in zip(regions['contour'], regions['bbox-0'], regions['bbox-1']):
        coord = np.squeeze(contours)
        coord = np.concatenate((coord, coord[0:1]), axis=0)
        ax[0].plot(coord[:, 0]+x, -(coord[:, 1]+y))
        cts.append(coord.shape[0])


    for ind, (y, x) in enumerate(zip(regions['bbox-0'], regions['bbox-1'])):
        contours = []
        for i in range(70):
            xi = regions[f'fft-{i}-0-0'][ind]
            yi = regions[f'fft-{i}-0-1'][ind]
            contours.append([xi, yi])
        coord = np.stack(contours, axis=0)
        coord = np.concatenate((coord, coord[0:1]), axis=0)
        ax[1].plot(coord[:, 0]+x, -(coord[:, 1]+y))
        
    plt.show()
    # assert(0)
    # segments = slic(img, n_segments=625,
    # compactness=10,
    # max_num_iter=10,
    # convert2lab=True,
    # enforce_connectivity=False,
    # slic_zero=False)

    # fig, ax = plt.subplots(1, 2)
    # labels = np.unique(segments)
    # ax[0].imshow(mark_boundaries(img, segments))
    # ax[0].set_title(f"Without enforcing connectivity: {len(labels)}")
    # ax[0].axis('off')
    # labels = np.unique(segments_ec)
    # ax[1].imshow(mark_boundaries(img, segments_ec))
    # ax[1].set_title(f"With enforcing connectivity: {len(labels)}")
    # ax[1].axis('off')

    # plt.show()
 

    print(np.min(cts), np.max(cts))
    assert(0)

    
