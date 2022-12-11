import numpy as np
from dataset.constants import *
import cv2
from util.util import *


def image_stdev(region, intensities):
    # note the ddof arg to get the sample var if you so desire!
    return np.std(intensities[region])


def hist(region, intensities):
    # note the ddof arg to get the sample var if you so desire!
    (hist, _) = np.histogram(intensities[region], bins=BINS, range=(0, 255), density=False)
    return hist


def polarize(region):
    # note the ddof arg to get the sample var if you so desire!
    centroid = np.mean(np.nonzero(region),axis=1)
    coords = np.nonzero(region)
    normalized_coords = np.stack([coords[0]-centroid[0], coords[1]-centroid[1]], axis=1)
    rho = np.linalg.norm(normalized_coords, axis=1)
    phi = np.arctan2(normalized_coords[:, 0], normalized_coords[:, 1])*180/np.pi+180
    radii_max = np.zeros([NUM_CHUNK, 2])
    # radii_min = np.zeros([NUM_CHUNK, 2])

    chunk = CHUNK
    
    for ind, degree in enumerate(range(0, 360, chunk)):
        try:
            radii_max[ind] = normalized_coords[np.argmax(np.where((degree<=phi) & (phi<degree+chunk), rho, np.zeros_like(rho)))]
        except: 
            pass
        
        # try:
        #     radii_min[ind] = normalized_coords[np.argmin(np.where((degree<=phi) & (phi<degree+chunk), rho, np.inf*np.ones_like(rho)))]
        # except: 
        #     pass
        
    return radii_max
    # return np.concatenate((radii_max, radii_min), axis=0)


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
    fourier_result = fourier_result[1:RESAMPLE_FFT_POINTS]

    amp = abs(fourier_result)
    phase = np.arctan2(fourier_result.imag, fourier_result.real)

    return np.concatenate((amp, phase))



def contours_euc(region):
    centroid = np.mean(np.nonzero(region),axis=1)
    region = (region*255).astype(np.uint8)
    contour, hierarchy = cv2.findContours(region, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    points = contour[0][:, 0, :]
    xi, yi = resample_2d(points, RESAMPLE_POINTS)
    contour_array = np.stack((xi, yi), axis=1)

    return contour_array-centroid

def contours_polar(region):
    centroid = np.mean(np.nonzero(region),axis=1)
    region = (region*255).astype(np.uint8)
    contour, hierarchy = cv2.findContours(region, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    points = contour[0][:, 0, :]
    xi, yi = resample_2d(points, RESAMPLE_POINTS)
    contour_array = np.stack((xi, yi), axis=1)-centroid

    rho = np.linalg.norm(contour_array, axis=1)
    phi = np.arctan2(contour_array[:, 0], contour_array[:, 1])*180/np.pi+180

    return np.stack((rho, phi), axis=1)