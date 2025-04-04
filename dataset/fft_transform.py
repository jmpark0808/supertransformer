import numpy as np
import math

def horizontal_flip(array, coeff, chance, size, resolution, seq_mask=None):
    if np.random.random() < chance:
        phase = array[:, 8+coeff:8+coeff+coeff]
        mask = phase > 0
        phase_flipped = np.empty_like(phase)
        phase_flipped[mask] = np.pi - phase[mask]
        phase_flipped[~mask] = -np.pi - phase[~mask]
        array[:, 8+coeff:8+coeff+coeff] = phase_flipped

        xs = array[:, 1]
        mid_x = size/2.
        diff_x = xs-mid_x
        array[:, 1] = mid_x-diff_x
        array = array.reshape(resolution[0], resolution[1], -1)
        array = np.fliplr(array)
        array = array.reshape(resolution[0]*resolution[1], -1)
        if seq_mask is not None:
            seq_mask = seq_mask.reshape(resolution[0], resolution[1])
            seq_mask= np.fliplr(seq_mask)
            seq_mask= seq_mask.reshape(-1)
    if seq_mask is not None:
        return array, seq_mask
    else:
        return array
    

    
def horizontal_flip_moments(centroids, colour, amp, phase, moments, lbp, chance, size, resolution, seq_mask=None):
    if np.random.random() < chance:
        # Flip centroids
        # xs = centroids[:, 1]
        centroids[:, 1] = -centroids[:, 1]
        # Flip moments
        moments = moments*np.array([1, -1, 1, 1, -1, 1, 1, -1]) 
        # LR all
        centroids = centroids.reshape(resolution[0], resolution[1], -1)
        centroids = np.fliplr(centroids)
        centroids = centroids.reshape(resolution[0]*resolution[1], -1)

        colour = colour.reshape(resolution[0], resolution[1], -1)
        colour = np.fliplr(colour)
        colour = colour.reshape(resolution[0]*resolution[1], -1)

        amp = amp.reshape(resolution[0], resolution[1], -1)
        amp = np.fliplr(amp)
        amp = amp.reshape(resolution[0]*resolution[1], -1)

        phase = phase.reshape(resolution[0], resolution[1], -1)
        phase = np.fliplr(phase)
        phase = phase.reshape(resolution[0]*resolution[1], -1)

        moments = moments.reshape(resolution[0], resolution[1], -1)
        moments = np.fliplr(moments)
        moments = moments.reshape(resolution[0]*resolution[1], -1)

        lbp = lbp.reshape(resolution[0], resolution[1], -1)
        lbp = np.fliplr(lbp)
        lbp = lbp.reshape(resolution[0]*resolution[1], -1)

        
        if seq_mask is not None:
            seq_mask = seq_mask.reshape(resolution[0], resolution[1])
            seq_mask= np.fliplr(seq_mask)
            seq_mask= seq_mask.reshape(-1)
    if seq_mask is not None:
        return centroids, colour, amp, phase, moments, lbp, seq_mask
    else:
        return centroids, colour, amp, phase, moments, lbp

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

def rotate(array, coeff, degrees, chance, size):
    if np.random.random() < chance:
        phase = array[:, 8+coeff:8+2*coeff]
        random_degrees = np.random.randint(-degrees, degrees)
        radians = math.radians(random_degrees)
        new_phase = phase + radians
        ys = np.sin(new_phase)
        xs = np.cos(new_phase)
        phase = np.arctan2(ys, xs)
        array[:, 8+coeff:8+2*coeff] = phase

        ys = array[:, 0]
        xs = array[:, 1]
        new_xs, new_ys = rotate_points([size[0]//2, size[1]//2], [xs, ys], radians)
        array[:, 0] = new_ys
        array[:, 1] = new_xs


    return array
# import matplotlib.pyplot as plt
# fig, ax = plt.subplots(1, 2)
# random_phase = np.random.random(70)*2*np.pi-np.pi

# ax[0].stem(random_phase)


# random_degrees = 20
# radians = math.radians(random_degrees)
# new_phase = random_phase + radians
# ys = np.sin(new_phase)
# xs = np.cos(new_phase)
# phase = np.arctan2(ys, xs)
# ax[1].stem(phase)
# plt.show()