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


def resample_2d(points, N):

    xc = points[:, 0].tolist() + [points[0, 0]]
    yc = points[:, 1].tolist() + [points[0, 1]]

    dx = np.diff(xc)
    dy = np.diff(yc)

    dS = np.sqrt(dx**2+dy**2)
    dS = np.array([0]+dS.tolist())

    d = np.cumsum(dS)

    perim = d[-1]

    ds = perim/N
    dSi = ds*np.arange(0, N)
    dSi[-1] = dSi[-1] - 0.005

    xi = np.interp(dSi, d, xc)
    yi = np.interp(dSi, d, yc)
    return xi, yi

def get_angle(points):
    # Compute centroid (mean of points)
    centroid = np.mean(points, axis=0)

    # Center the points around the centroid
    centered_points = points - centroid

    # Compute the covariance matrix
    cov_matrix = np.cov(centered_points, rowvar=False)

    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # The eigenvector with the largest eigenvalue is the principal axis
    major_axis = eigenvectors[:, 1]  # Eigenvector corresponding to the largest eigenvalue
    minor_axis = eigenvectors[:, 0]
    
    
    # Compute the orientation angle (in degrees)
    angle_major = math.degrees(np.arctan2(major_axis[1], major_axis[0]))
    angle_minor = math.degrees(np.arctan2(minor_axis[1], minor_axis[0]))

    return angle_major, angle_minor, np.sqrt(eigenvalues[1]), np.sqrt(eigenvalues[0])


N = 70

xi, yi = resample_2d(points, N)
# ax[1].scatter(xi, yi)
# plt.show()
fig, ax = plt.subplots(3, 4)
contour_array = np.stack((xi, yi), axis=1)


contour_complex = np.empty(contour_array.shape[:-1], dtype=complex)
contour_complex.real = contour_array[:, 0]
contour_complex.imag = contour_array[:, 1]
fourier_result = np.fft.fft(contour_complex)


if np.abs(fourier_result[1]) > 1e-6:
    ref_phase = np.angle(fourier_result[1])
else:
    ref_phase = 0.0
    
# Normalize phases: subtract reference phase from each coefficient
fourier_result = np.abs(fourier_result) * np.exp(1j * (np.angle(fourier_result) - ref_phase))
phase = np.angle(fourier_result)


# Align phases relative to this component


ax[0, 0].scatter(xi[2:], yi[2:], c='blue')
ax[0, 0].scatter(xi[0], yi[0], c='red', label='Start')
ax[0, 0].scatter(xi[1], yi[1], c='Green', label='Second')
ax[0, 0].legend()
ax[0, 0].set_title('Original')
ax[0, 0].set_ylabel('Images')
ax[0, 0].set_aspect('equal')

ax[1, 0].stem(np.linspace(0, np.pi, len(fourier_result))[1:], abs(fourier_result[1:]), 'b', markerfmt=" ", basefmt="-b")
ax[1, 0].set_ylabel('Amplitude')


ax[2, 0].stem(np.linspace(0, np.pi, len(fourier_result))[1:], phase[1:], 'b', markerfmt=" ", basefmt="-b")
# ax[2, 0].stem([0], [phase])
# circ = plt.Circle((0, 0), radius=1, edgecolor='b', facecolor='None')
# ax[2,0].add_patch(circ)
# ax[2,0].plot([0, x_major], [0, y_major])
# ax[2,0].plot([0, x_minor], [0, y_minor])
# ellipse = Ellipse([0,0], length_major*2, length_minor*2, angle_major, facecolor='none', edgecolor='red')
# ax[2, 0].add_patch(ellipse)
ax[2, 0].set_ylabel('Phase')



# Changing starting point
xi_roll, yi_roll = np.roll(xi, 5), np.roll(yi, 5)

contour_array = np.stack((xi_roll, yi_roll), axis=1)
contour_complex = np.empty(contour_array.shape[:-1], dtype=complex)
contour_complex.real = contour_array[:, 0]
contour_complex.imag = contour_array[:, 1]
fourier_result = np.fft.fft(contour_complex)


# x_, y_, major, minor = get_angle(contour_array)
if np.abs(fourier_result[1]) > 1e-6:
    ref_phase = np.angle(fourier_result[1])
else:
    ref_phase = 0.0
    
# Normalize phases: subtract reference phase from each coefficient
fourier_result = np.abs(fourier_result) * np.exp(1j * (np.angle(fourier_result) - ref_phase))
phase = np.angle(fourier_result)

ax[0, 1].scatter(xi_roll[2:], yi_roll[2:], c='blue')
ax[0, 1].scatter(xi_roll[0], yi_roll[0], c='red', label='Start')
ax[0, 1].scatter(xi_roll[1], yi_roll[1], c='Green', label='Second')
ax[0, 1].legend()
ax[0, 1].set_title('New starting point')
ax[0, 1].set_aspect('equal')
ax[1, 1].stem(np.linspace(0, np.pi, len(fourier_result))[1:], abs(fourier_result[1:]), 'b', markerfmt=" ", basefmt="-b")



ax[2, 1].stem(np.linspace(0, np.pi, len(fourier_result))[1:], phase[1:], 'b', markerfmt=" ", basefmt="-b")
# ax[2,1].stem([0], [phase])
# circ = plt.Circle((0, 0), radius=1, edgecolor='b', facecolor='None')
# ax[2,1].add_patch(circ)
# ax[2,1].plot([0, major[0]], [0, major[1]])
# ax[2,1].plot([0, minor[0]], [0, minor[1]])

# ROTATION

xc = np.mean(xi)
yc = np.mean(yi)

xi_rotate, yi_rotate = rotate([xc, yc], [xi, yi], math.radians(90))
contour_array = np.stack((xi_rotate, yi_rotate), axis=1)
contour_complex = np.empty(contour_array.shape[:-1], dtype=complex)
contour_complex.real = contour_array[:, 0]
contour_complex.imag = contour_array[:, 1]
fourier_result = np.fft.fft(contour_complex)


# angle_major, angle_minor, length_major, length_minor = get_angle(contour_array)
# x_major, y_major = solve_triangle(length_major, angle_major)
# x_minor, y_minor = solve_triangle(length_minor, angle_minor)
if np.abs(fourier_result[1]) > 1e-6:
    ref_phase = np.angle(fourier_result[1])
else:
    ref_phase = 0.0
    
# Normalize phases: subtract reference phase from each coefficient
fourier_result = np.abs(fourier_result) * np.exp(1j * (np.angle(fourier_result) - ref_phase))
phase = np.angle(fourier_result)

ax[0, 2].scatter(xi_rotate[2:], yi_rotate[2:], c='blue')
ax[0, 2].scatter(xi_rotate[0], yi_rotate[0], c='red', label='Start')
ax[0, 2].scatter(xi_rotate[1], yi_rotate[1], c='Green', label='Second')
ax[0, 2].legend()
ax[0, 2].set_title('Rotated')
ax[0, 2].set_aspect('equal')
ax[1, 2].stem(np.linspace(0, np.pi, len(fourier_result))[1:], abs(fourier_result[1:]), 'b', markerfmt=" ", basefmt="-b")
# ax[1, 3].plot(fourier_result.real[1:], label='Real')
# ax[1, 3].plot(fourier_result.imag[1:], label='Imag')
# ax[1, 3].legend(loc='lower left')

# circ = plt.Circle((0, 0), radius=1, edgecolor='b', facecolor='None')
# ax[2,2].add_patch(circ)
# ax[2,2].plot([0, major[0]], [0, major[1]])
# ax[2,2].plot([0, minor[0]], [0, minor[1]])
# ax[2,2].plot([0, x_major], [0, y_major])
# ax[2,2].plot([0, x_minor], [0, y_minor])
# ellipse = Ellipse([0,0], length_major*2, length_minor*2, angle_major, facecolor='none', edgecolor='red')
# ax[2, 2].add_patch(ellipse)
ax[2, 2].stem(np.linspace(0, np.pi, len(fourier_result))[1:], phase[1:], 'b', markerfmt=" ", basefmt="-b")
# ax[2,2].stem([0], [phase])


# Horizontal Flip


xi_flip = np.roll(np.flip(-xi), 1)#horizontal_flip(xi)
yi_flip = np.roll(np.flip(yi), 1)
contour_array = np.stack((xi_flip, yi_flip), axis=1)
contour_complex = np.empty(contour_array.shape[:-1], dtype=complex)
contour_complex.real = contour_array[:, 0]
contour_complex.imag = contour_array[:, 1]
fourier_result = np.fft.fft(contour_complex)

if np.abs(fourier_result[1]) > 1e-6:
    ref_phase = np.angle(fourier_result[1])
else:
    ref_phase = 0.0
    
# Normalize phases: subtract reference phase from each coefficient
fourier_result = np.abs(fourier_result) * np.exp(1j * (np.angle(fourier_result) - ref_phase))
phase = np.angle(fourier_result)

angle_major, angle_minor, length_major, length_minor = get_angle(contour_array)
x_major, y_major = solve_triangle(length_major, angle_major)
x_minor, y_minor = solve_triangle(length_minor, angle_minor)
ax[0, 3].scatter(xi_flip[2:], yi_flip[2:], c='blue')
ax[0, 3].scatter(xi_flip[0], yi_flip[0], c='red', label='Start')
ax[0, 3].scatter(xi_flip[1], yi_flip[1], c='Green', label='Second')
ax[0, 3].legend()
ax[0, 3].set_title('Horizontal Flipped')
ax[0, 3].set_aspect('equal')
ax[1, 3].stem(np.linspace(0, np.pi, len(fourier_result))[1:], abs(fourier_result[1:]), 'b', markerfmt=" ", basefmt="-b")
# ax[1, 3].plot(fourier_result.real[1:], label='Real')
# ax[1, 3].plot(fourier_result.imag[1:], label='Imag')
# ax[1, 3].legend(loc='lower left')
ax[2, 3].stem(np.linspace(0, np.pi, len(fourier_result))[1:], phase[1:], 'b', markerfmt=" ", basefmt="-b")
# circ = plt.Circle((0, 0), radius=1, edgecolor='b', facecolor='None')
# ax[2,3].add_patch(circ)
# ax[2,3].plot([0, major[0]], [0, major[1]])
# ax[2,3].plot([0, minor[0]], [0, minor[1]])
# ax[2,3].plot([0, x_major], [0, y_major])
# ax[2,3].plot([0, x_minor], [0, y_minor])
# ellipse = Ellipse([0,0], length_major*2, length_minor*2, angle_major, facecolor='none', edgecolor='red')
# ax[2, 3].add_patch(ellipse)




fig.supxlabel('Frequency (2nd and 3rd row)')
plt.show()
