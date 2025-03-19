import numpy as np
import math
def rotate_points(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point
    zeros = np.argwhere(np.logical_and(px == 0 , py == 0))
    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    qx[zeros] = 0
    qy[zeros] = 0
    return qx, qy



def rotate_moments(moments, centroids, prob, rotation_angle_degrees, size):
    if np.random.random() < prob:
        random_degrees = np.random.randint(-rotation_angle_degrees, rotation_angle_degrees)
        theta = math.radians(random_degrees)


        ys = centroids[:, 0]
        xs = centroids[:, 1]
        new_xs, new_ys = rotate_points([size[0]//2, size[1]//2], [xs, ys], theta)
        centroids[:, 0] = new_ys
        centroids[:, 1] = new_xs
    
        # Extract second-order moments
        mu20 = moments[:, 2]
        mu02 = moments[:, 3]
        mu11 = moments[:, 1]
        
        # Rotate second-order moments
        mu20_rot = mu20 * (np.cos(theta)**2) + mu02 * (np.sin(theta)**2) - 2 * mu11 * np.sin(theta) * np.cos(theta)
        mu02_rot = mu20 * (np.sin(theta)**2) + mu02 * (np.cos(theta)**2) + 2 * mu11 * np.sin(theta) * np.cos(theta)
        mu11_rot = (mu20 - mu02) * np.sin(theta) * np.cos(theta) + mu11 * (np.cos(theta)**2 - np.sin(theta)**2)
        
        # Extract third-order moments
        mu30 = moments[:, 6]
        mu03 = moments[:, 7]
        mu21 = moments[:, 4]
        mu12 = moments[:, 5]
            
        # Rotate third-order moments
        mu30_rot = (mu30 * np.cos(theta)**3 
                    - 3 * mu21 * np.cos(theta)**2 * np.sin(theta)
                    + 3 * mu12 * np.cos(theta) * np.sin(theta)**2 
                    - mu03 * np.sin(theta)**3)
        
        mu03_rot = (mu30 * np.sin(theta)**3 
                    + 3 * mu21 * np.cos(theta) * np.sin(theta)**2
                    + 3 * mu12 * np.cos(theta)**2 * np.sin(theta) 
                    + mu03 * np.cos(theta)**3)
        
        mu21_rot = (mu30 * np.cos(theta)**2 * np.sin(theta)
                    + mu21 * (np.cos(theta)**3 - 2*np.cos(theta)*np.sin(theta)**2)
                    + mu12 * (np.sin(theta)**3 - 2*np.cos(theta)**2*np.sin(theta))
                    + mu03 * np.cos(theta)*np.sin(theta)**2)
        
        mu12_rot = (mu30 * np.cos(theta) * np.sin(theta)**2
                    + mu21 * (2*np.cos(theta)**2*np.sin(theta) - np.sin(theta)**3)
                    + mu12 * (np.cos(theta)**3 - 2*np.cos(theta)*np.sin(theta)**2)
                    - mu03 * np.cos(theta)**2 * np.sin(theta))
        
        
        rotated_moments = np.copy(moments)
        rotated_moments[:, 2] = mu20_rot
        rotated_moments[:, 3] = mu02_rot
        rotated_moments[:, 1] = mu11_rot

        rotated_moments[:, 6] = mu30_rot
        rotated_moments[:, 7] = mu03_rot
        rotated_moments[:, 4] = mu21_rot
        rotated_moments[:, 5] = mu12_rot
        return rotated_moments, centroids
    else:
        return moments, centroids
    

def flip_moments(moments, prob):
    moments_ = np.copy(moments)
    if np.random.random() < prob:
        moments_ = moments_*np.array([1, -1, 1, 1, -1, 1, 1, -1]) 
    return moments_

def log_moments(moments):
    moments_ = np.sign(moments)*np.where(moments == 0, 0, np.log(np.abs(moments)))
    return moments_ # Return the central moments of the largest region

