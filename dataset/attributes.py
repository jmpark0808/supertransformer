import numpy as np
from dataset.constants import *
import cv2
from util.util import *
import math
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt

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
			bins=np.arange(0, LBP_POINTS*LBP_RADIUS+3),
			range=(0, LBP_POINTS*LBP_RADIUS+2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    return hist




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


def eccen(region):
    centroid = np.mean(np.nonzero(region),axis=1)
    centroid_x = centroid[1]
    centroid_y = -centroid[0]
    coords = np.nonzero(region)
    coords_x = coords[1]
    coords_y = -coords[0]

    region_shape = region.shape
    




    # sigma = np.matmul(coords-centroid[:, None], (coords-centroid[:, None]).T)/len(coords)
    if (coords-centroid[:, None]).shape[1]==1:
        return np.array([0, 0, 0])
    U, S, V = np.linalg.svd(np.stack((coords_x-centroid_x, coords_y-centroid_y)))
    # U, S = np.linalg.eig(coords-centroid[:, None])

    angle = np.arctan2(U[1],U[0])

    major_angle = angle[0]
    if major_angle < 0:
        major_angle += np.pi


    tt = np.linspace(0, 2*np.pi, 1000)
    circle = np.stack((np.cos(tt), np.sin(tt)))    # unit circle
    transform = np.sqrt(2/len(coords[0])) * U.dot(np.diag(S))   # transformation matrix
    fit = transform.dot(circle) + np.array([[centroid_x], [centroid_y]])


    # Check square
    if np.prod(region_shape) == len(coords[0]):
        if region_shape[1]>= region_shape[0]: # longer horizontally
            major_angle = 0
        else:
            major_angle = np.pi/2
        # plt.plot(coords_x, coords_y,  '.')
        # plt.plot(fit[0, :], fit[1, :],  'r')
        # plt.plot([centroid_x, centroid_x+np.sqrt(2/len(coords[0]))*S[0]*math.cos(major_angle)], 
        #          [centroid_y, centroid_y+np.sqrt(2/len(coords[0]))*S[0]*math.sin(major_angle)])
        # plt.title(f'{np.sqrt(2/len(coords[0]))*S[0]}, {np.sqrt(2/len(coords[0]))*S[1]}, {major_angle}, {region_shape}')
        # plt.axis('scaled')
        # plt.show()

    return np.array([np.sqrt(2/len(coords[0]))*S[0], np.sqrt(2/len(coords[0]))*S[1], major_angle])

    


