import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import math
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

contour, hierarchy = cv2.findContours(rectangle, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
points = contour[0][:, 0, :]
# ax[0].scatter(points[:,0], points[:, 1])

distances = []
for i in range(len(points)):
    if i == len(points)-1:
        distances.append(euc_distance(points[i], points[0]))
    else:
        distances.append(euc_distance(points[i], points[i+1]))

# plt.plot(distances)
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
    
    orientation = region.orientation
    y0, x0 = region.centroid
    x1 = x0 + np.cos(orientation) * 0.5 * minor_axis_length
    y1 = y0 - np.sin(orientation) * 0.5 * minor_axis_length

    x2 = x0 - np.sin(orientation) * 0.5 * major_axis_length
    y2 = y0 - np.cos(orientation) * 0.5 * major_axis_length

    return math.degrees(orientation), x1, y1, x2, y2, x0, y0


def get_pca_rotation(image):
     # Convert contour to (N, 2) shape
    contour = np.array(np.nonzero(image)).transpose((1,0))
    
    # Compute the mean (centroid)
    mean = np.mean(contour, axis=0)

    # Center the points
    centered = contour - mean

    # Compute the covariance matrix and get eigenvectors
    cov_matrix = np.cov(centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)  # Ensures sorted order

    # The principal eigenvector (largest eigenvalue)
    principal_vector = eigenvectors[:, 1]  # Last column is the major axis

    # Compute the angle with respect to the x-axis
    angle_rad = np.arctan2(principal_vector[1], principal_vector[0])
    angle_deg = np.degrees(angle_rad)

    # Ensure the angle is in [0, 360)
    # if angle_deg < 0:
    #     angle_deg += 360

    y0, x0 = mean
    orientation = math.radians(angle_deg)
    minor_axis_length = eigenvalues[0]
    major_axis_length = eigenvalues[1]
    x1 = x0 + np.cos(orientation) * 0.5 * minor_axis_length
    y1 = y0 - np.sin(orientation) * 0.5 * minor_axis_length

    x2 = x0 - np.sin(orientation) * 0.5 * major_axis_length
    y2 = y0 - np.cos(orientation) * 0.5 * major_axis_length

    return angle_deg, x1, y1, x2, y2, x0, y0


for i in range(0, 360, 10):
    img = Image.fromarray(rectangle)
    img = img.rotate(i)
    img_np = np.array(img)
    degrees, x1, y1, x2, y2, x0, y0 = get_pca_rotation(img_np)


    plt.imshow(img)
    plt.plot((x0, x1), (y0, y1), '-r', linewidth=2.5)
    plt.plot((x0, x2), (y0, y2), '-b', linewidth=2.5)
    plt.plot(x0, y0, '.g', markersize=15)
    plt.savefig(f'/home/eddie/Downloads/gif/{i}')
    plt.clf()
   