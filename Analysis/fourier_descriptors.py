import os
import cv2
import numpy as np
from pyefd import elliptic_fourier_descriptors
from PIL import Image
import  matplotlib.pyplot as plt

def truncate_descriptor(descriptors, degree):
    """this function truncates an unshifted fourier descriptor array
    and returns one also unshifted"""

    descriptors = np.fft.fftshift(descriptors)
    center_index = len(descriptors) // 2
    descriptors = descriptors[center_index - degree // 2:center_index + degree // 2]
    descriptors = np.fft.ifftshift(descriptors)
    return descriptors


def reconstruct(descriptors, degree):
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
    if contour_reconstruct.min() < 0:
        contour_reconstruct -= contour_reconstruct.min()
    # normalization
    contour_reconstruct *= 800 / contour_reconstruct.max()
    # type cast to int32
    contour_reconstruct = contour_reconstruct.astype(np.int32, copy=False)
    black = np.zeros((800, 800), np.uint8)
    # draw and visualize
    cv2.drawContours(black, contour_reconstruct, -1, 255, thickness=-1)
    cv2.imshow("black", black)
    cv2.waitKey(1000)
    cv2.imwrite("reconstruct_result.jpg", black)
    cv2.destroyAllWindows()
    return descriptor_in_use

mask_dir = '/mnt/hdd/Datasets/DUTS/DUTS-TE/Mask'

for mask in os.listdir(mask_dir):
    maskpath = os.path.join(mask_dir, mask)
    image = cv2.imread(maskpath)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray[gray<=125] = 0
    gray[gray>125] = 255
    contour, hierarchy = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contour_array = contour[0][:, 0, :]
    contour_complex = np.empty(contour_array.shape[:-1], dtype=complex)
    contour_complex.real = contour_array[:, 0]
    contour_complex.imag = contour_array[:, 1]
    fourier_result = np.fft.fft(contour_complex)
    print(np.absolute(fourier_result))
    assert(0)
    contour_reconstruct = truncate_descriptor(fourier_result, 18)
    print(len(contour_reconstruct))