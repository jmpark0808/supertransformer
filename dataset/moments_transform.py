import numpy as np

def rotate_moments(moments, prob, rotation_angle_degrees):
    if np.random.random() < prob:
        random_degrees = np.random.randint(-rotation_angle_degrees, rotation_angle_degrees)
        theta = np.deg2rad(random_degrees)
    
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
        return rotated_moments
    else:
        return moments
    

def flip_moments(moments, prob):
    moments_ = np.copy(moments)
    if np.random.random() < prob:
        moments_ = moments_*np.array([1, -1, 1, 1, -1, 1, 1, -1]) 
    return moments_

def log_moments(moments):
    moments_ = np.sign(moments)*np.log(np.abs(moments)+1e-10)
    return moments_ # Return the central moments of the largest region

