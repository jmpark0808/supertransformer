import numpy as np
import math

def horizontal_flip(array, coeff, chance):
    if np.random.random() < chance:
        phase = array[:, -coeff:]
        mask = phase > 0
        phase_flipped = np.empty_like(phase)
        phase_flipped[mask] = np.pi - phase[mask]
        phase_flipped[~mask] = -np.pi - phase[~mask]
        array[:, -coeff:] = phase_flipped

        xs = array[:, 1]
        mid_x = 300/2.
        diff_x = xs-mid_x
        array[:, 1] = mid_x-diff_x
    return array

def rotate_points(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy

def rotate(array, coeff, degrees, chance):
    if np.random.random() < chance:
        phase = array[:, -(coeff*2):]
        random_degrees = np.random.randint(-degrees, degrees)
        radians = math.radians(random_degrees)
        new_phase = phase + radians
        ys = np.sin(new_phase)
        xs = np.cos(new_phase)
        phase = np.arctan2(ys, xs)
        array[:, -(coeff*2):] = phase

        ys = array[:, 0]
        xs = array[:, 1]
        new_xs, new_ys = rotate_points([128, 128], [xs, ys], radians)
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