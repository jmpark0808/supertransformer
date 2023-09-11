import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image

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
fig, ax = plt.subplots(1, 1)
contour_array = np.stack((xi, yi), axis=1)
# plt.scatter(points[1:, 0], points[1:, 1], c='blue')
# plt.scatter(points[0, 0], points[0, 1], c='red')
# plt.show()

contour_complex = np.empty(contour_array.shape[:-1], dtype=complex)
contour_complex.real = contour_array[:, 0]
contour_complex.imag = contour_array[:, 1]
fourier_result = np.fft.fft(contour_complex)
ax.scatter(xi[2:], yi[2:], c='purple', label='original')
ax.scatter(xi[0], yi[0], c='red', label='Start')
ax.scatter(xi[1], yi[1], c='Green', label='Second')
ax.legend()
ax.set_title('Original')




# Use one coefficients 
xi, yi = resample_2d(points, N)

contour_complex = np.empty(contour_array.shape[:-1], dtype=complex)
contour_complex.real = contour_array[:, 0]
contour_complex.imag = contour_array[:, 1]
fourier_result = np.fft.fft(contour_complex)

truncated_fourier_result = np.copy(fourier_result)
truncated_fourier_result[1:-1] = 0
inverse_fourier_result = np.fft.ifft(truncated_fourier_result)
contour_reconstruct = np.array(
        [inverse_fourier_result.real, inverse_fourier_result.imag])
contour_reconstruct = np.transpose(contour_reconstruct)

xi = contour_reconstruct[:, 0]
yi = contour_reconstruct[:, 1]
ax.scatter(xi[2:], yi[2:], c='blue', label='1 coeff (last)')
ax.scatter(xi[0], yi[0], c='red')
ax.scatter(xi[1], yi[1], c='Green')
ax.legend()
ax.set_title('Use 1 coefficient')

# Set coefficient has 500 for first two and last two
xi, yi = resample_2d(points, N)

contour_complex = np.empty(contour_array.shape[:-1], dtype=complex)
contour_complex.real = contour_array[:, 0]
contour_complex.imag = contour_array[:, 1]
fourier_result = np.fft.fft(contour_complex)

truncated_fourier_result = np.copy(fourier_result)
truncated_fourier_result[1:-1] = 0
truncated_fourier_result[1:3] = 500
truncated_fourier_result[-2:] = 500
inverse_fourier_result = np.fft.ifft(truncated_fourier_result)
contour_reconstruct = np.array(
        [inverse_fourier_result.real, inverse_fourier_result.imag])
contour_reconstruct = np.transpose(contour_reconstruct)

xi = contour_reconstruct[:, 0]
yi = contour_reconstruct[:, 1]
ax.scatter(xi[2:], yi[2:], c='orange', label='4 coeff (first two and last two as 500)')
ax.scatter(xi[0], yi[0], c='red')
ax.scatter(xi[1], yi[1], c='Green')
ax.legend()
ax.set_title('Use 4 coef of 500')


plt.show()
