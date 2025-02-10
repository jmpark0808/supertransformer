import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import math

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




N = 70

xi, yi = resample_2d(points, N)
# ax[1].scatter(xi, yi)
# plt.show()
fig, ax = plt.subplots(1, 3)
contour_array = np.stack((xi, yi), axis=1)
# plt.scatter(points[1:, 0], points[1:, 1], c='blue')
# plt.scatter(points[0, 0], points[0, 1], c='red')
# plt.show()

contour_complex = np.empty(contour_array.shape[:-1], dtype=complex)
contour_complex.real = contour_array[:, 0]
contour_complex.imag = contour_array[:, 1]
fourier_result = np.fft.fft(contour_complex)
ax[0].scatter(xi[2:], yi[2:], c='blue')
ax[0].scatter(xi[0], yi[0], c='red', label='Start')
ax[0].scatter(xi[1], yi[1], c='Green', label='Second')
ax[0].legend()
ax[0].set_title('Original')
ax[0].set_ylabel('Images')
ax[0].set_aspect('equal')

amplitude = abs(fourier_result)
phase = np.arctan2(fourier_result.imag, fourier_result.real)
# ax[1].stem(np.linspace(0, np.pi, len(fourier_result))[1:], phase[1:], 'b', markerfmt=" ", basefmt="-b")
# ax[1].set_ylabel('Phase')

radians = math.radians(90)
new_phase = phase + radians
ys = np.sin(new_phase)
xs = np.cos(new_phase)
phase_rotated = np.arctan2(ys, xs) 


ifft = amplitude * np.exp(1j * phase_rotated)
ifft = np.fft.ifft(ifft)
ifft = np.array(
        [ifft.real, ifft.imag])
ifft = np.transpose(ifft)

xi_r, yi_r = ifft[:, 0], ifft[:, 1]

ax[2].scatter(xi_r[2:], yi_r[2:], c='blue')
ax[2].scatter(xi_r[0], yi_r[0], c='red', label='Start')
ax[2].scatter(xi_r[1], yi_r[1], c='Green', label='Second')
ax[2].legend()
ax[2].set_aspect('equal')

# ROTATION

xc = np.mean(xi)
yc = np.mean(yi)

xi_rotate, yi_rotate = rotate([xc, yc], [xi, yi], math.radians(90))
contour_array = np.stack((xi_rotate, yi_rotate), axis=1)
contour_complex = np.empty(contour_array.shape[:-1], dtype=complex)
contour_complex.real = contour_array[:, 0]
contour_complex.imag = contour_array[:, 1]
fourier_result = np.fft.fft(contour_complex)
ax[1].scatter(xi_rotate[2:], yi_rotate[2:], c='blue')
ax[1].scatter(xi_rotate[0], yi_rotate[0], c='red', label='Start')
ax[1].scatter(xi_rotate[1], yi_rotate[1], c='Green', label='Second')
ax[1].legend()
ax[1].set_title('Rotated')
ax[1].set_aspect('equal')
# ax[1, 2].stem(np.linspace(0, np.pi, len(fourier_result))[1:], abs(fourier_result[1:]), 'b', markerfmt=" ", basefmt="-b")


# phase = np.arctan2(fourier_result.imag, fourier_result.real)
# ax[2, 2].stem(np.linspace(0, np.pi, len(fourier_result))[1:], phase[1:], 'b', markerfmt=" ", basefmt="-b")



fig.supxlabel('Frequency (2nd and 3rd row)')
plt.show()


