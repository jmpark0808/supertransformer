import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import math
from matplotlib.patches import Ellipse
from skimage.measure import regionprops, label

def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy

def horizontal_flip(xs):
    mid_x = np.mean(xs)
    diff_x = xs-mid_x
    new_x = mid_x-diff_x
    return new_x


def solve_triangle(hypotenuse, angle_degrees):
    # Convert angle to radians
    angle_radians = math.radians(angle_degrees)
    
    # Calculate adjacent side (x)
    x = hypotenuse * math.cos(angle_radians)
    
    # Calculate opposite side (y)
    y = hypotenuse * math.sin(angle_radians)
    
    return x, y


# rectangle = np.zeros([100, 100])
# rectangle[30:70, 40:60] = 1
# rectangle[25:30, 55:60] = 1
# rectangle[35:40, 35:40] = 1
# rectangle = (rectangle*255).astype(np.uint8)

rectangle = np.load('/home/eddie/waterloo/supertransformer/Analysis/sample_sp.npy')
rectangle = (rectangle*255).astype(np.uint8)
# fig, ax = plt.subplots(1, 2)
# ax[0].imshow(rectangle, cmap='gray', origin='lower')
# ax[1].imshow(rectangle, cmap='gray', origin='lower')
# plt.imshow(rectangle, cmap='gray')
# plt.show()
# assert(0)
def euc_distance(pt1, pt2):
    y_diff = np.abs(pt2[1]-pt1[1])
    x_diff = np.abs(pt2[0]-pt1[0])
    return np.sqrt(y_diff**2+x_diff**2)




N = 70


# ax[1].scatter(xi, yi)
# plt.show()
def get_rotation(image):

    labeled_image = label(image)
    regions = regionprops(labeled_image)

    # Get the first (and only) region
    region = regions[0]

    # Extract major and minor axis lengths
    major_axis_length = region.major_axis_length
    minor_axis_length = region.minor_axis_length

    # print(major_axis_length)
    # print(minor_axis_length)
    # assert(0)
    # Calculate orientation (angle of the major axis)
    orientation = math.radians(region.orientation + 90)

    # Calculate endpoints of major and minor axes
    y0, x0 = region.centroid
    x1 = x0 + np.cos(orientation) * 0.5 * major_axis_length
    y1 = y0 - np.sin(orientation) * 0.5 * major_axis_length

    x2 = x0 - np.sin(orientation) * 0.5 * minor_axis_length
    y2 = y0 - np.cos(orientation) * 0.5 * minor_axis_length

fig, ax = plt.subplots(1, 3)
