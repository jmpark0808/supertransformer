import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image

rectangle = np.zeros([100, 100])
rectangle[30:70, 40:60] = 1
rectangle[25:30, 55:60] = 1
rectangle[35:40, 35:40] = 1
rectangle = (rectangle*255).astype(np.uint8)





# plt.imshow(rectangle, cmap='gray')
# plt.show()

def euc_distance(pt1, pt2):
    y_diff = np.abs(pt2[1]-pt1[1])
    x_diff = np.abs(pt2[0]-pt1[0])
    return np.sqrt(y_diff**2+x_diff**2)


contour, hierarchy = cv2.findContours(rectangle, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
points = contour[0][:, 0, :]

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


fig, ax = plt.subplots(3, 6)

N = 70

xi, yi = resample_2d(points, N)
contour_array = np.stack((xi, yi), axis=1)
# plt.scatter(points[1:, 0], points[1:, 1], c='blue')
# plt.scatter(points[0, 0], points[0, 1], c='red')
# plt.show()

contour_complex = np.empty(contour_array.shape[:-1], dtype=complex)
contour_complex.real = contour_array[:, 0]
contour_complex.imag = contour_array[:, 1]
fourier_result = np.fft.fft(contour_complex)
ax[0, 0].scatter(xi[2:], yi[2:], c='blue')
ax[0, 0].scatter(xi[0], yi[0], c='red', label='Start')
ax[0, 0].scatter(xi[1], yi[1], c='Green', label='Second')
ax[0, 0].legend()
ax[0, 0].set_title('Original')
ax[0, 0].set_ylabel('Images')

ax[1, 0].stem(list(range(len(fourier_result[1:]))), abs(fourier_result[1:]), 'b', markerfmt=" ", basefmt="-b")
# ax[1, 0].plot(fourier_result.real[1:], label='Real')
# ax[1, 0].plot(fourier_result.imag[1:], label='Imag')
# ax[1, 0].legend(loc='lower left')
ax[1, 0].set_ylabel('Amplitude')

phase = np.arctan2(fourier_result.imag, fourier_result.real)
ax[2, 0].stem(list(range(len(phase[1:]))), phase[1:], 'b', markerfmt=" ", basefmt="-b")
ax[2, 0].set_ylabel('Phase')
print('Original', fourier_result.real[1:])
# plt.scatter(xi[1:], yi[1:], c='blue')
# plt.scatter(xi[0], yi[0], c='red')
# plt.show()


# TRANSLATION
contour_array = np.stack((xi, yi), axis=1)+50
contour_complex = np.empty(contour_array.shape[:-1], dtype=complex)
contour_complex.real = contour_array[:, 0]
contour_complex.imag = contour_array[:, 1]
fourier_result = np.fft.fft(contour_complex)

ax[0, 1].scatter(xi[2:]+50, yi[2:]+50, c='blue')
ax[0, 1].scatter(xi[0]+50, yi[0]+50, c='red', label='Start')
ax[0, 1].scatter(xi[1]+50, yi[1]+50, c='Green', label='Second')
ax[0, 1].legend()
ax[0, 1].set_title('Translated')

ax[1, 1].stem(list(range(len(fourier_result[1:]))), abs(fourier_result[1:]), 'b', markerfmt=" ", basefmt="-b")
# ax[1, 1].plot(fourier_result.real[1:], label='Real')
# ax[1, 1].plot(fourier_result.imag[1:], label='Imag')
# ax[1, 1].legend(loc='lower left')

phase = np.arctan2(fourier_result.imag, fourier_result.real)
ax[2, 1].stem(list(range(len(phase[1:]))), phase[1:], 'b', markerfmt=" ", basefmt="-b")

print('Translated', fourier_result.real[1:])
# SCALED
contour_array = np.stack((xi*10-315, yi*10-225), axis=1)
contour_complex = np.empty(contour_array.shape[:-1], dtype=complex)
contour_complex.real = contour_array[:, 0]
contour_complex.imag = contour_array[:, 1]
fourier_result = np.fft.fft(contour_complex)
ax[0, 2].scatter(xi[2:]*10-315, yi[2:]*10-225, c='blue')
ax[0, 2].scatter(xi[0]*10-315, yi[0]*10-225, c='red', label='Start')
ax[0, 2].scatter(xi[1]*10-315, yi[1]*10-225, c='Green', label='Second')
ax[0, 2].legend()
ax[0, 2].set_title('Scaled x 10')

ax[1, 2].stem(list(range(len(fourier_result[1:]))), abs(fourier_result[1:]), 'b', markerfmt=" ", basefmt="-b")
# ax[1, 2].plot(fourier_result.real[1:], label='Real')
# ax[1, 2].plot(fourier_result.imag[1:], label='Imag')
# ax[1, 2].legend(loc='lower left')

phase = np.arctan2(fourier_result.imag, fourier_result.real)
ax[2, 2].stem(list(range(len(phase[1:]))), phase[1:], 'b', markerfmt=" ", basefmt="-b")
print('Scaled', fourier_result.real[1:])
# ROTATION

im = Image.fromarray(rectangle)
rotated = im.rotate(30)
rectangle_rotated = np.array(rotated)

contour, hierarchy = cv2.findContours(rectangle_rotated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
points_rotated = contour[0][:, 0, :]

xi, yi = resample_2d(points_rotated, N)
contour_array = np.stack((xi, yi), axis=1)
contour_complex = np.empty(contour_array.shape[:-1], dtype=complex)
contour_complex.real = contour_array[:, 0]
contour_complex.imag = contour_array[:, 1]
fourier_result = np.fft.fft(contour_complex)
ax[0, 3].scatter(xi[2:], yi[2:], c='blue')
ax[0, 3].scatter(xi[0], yi[0], c='red', label='Start')
ax[0, 3].scatter(xi[1], yi[1], c='Green', label='Second')
ax[0, 3].legend()
ax[0, 3].set_title('Rotated')

ax[1, 3].stem(list(range(len(fourier_result[1:]))), abs(fourier_result[1:]), 'b', markerfmt=" ", basefmt="-b")
# ax[1, 3].plot(fourier_result.real[1:], label='Real')
# ax[1, 3].plot(fourier_result.imag[1:], label='Imag')
# ax[1, 3].legend(loc='lower left')

phase = np.arctan2(fourier_result.imag, fourier_result.real)
ax[2, 3].stem(list(range(len(phase[1:]))), phase[1:], 'b', markerfmt=" ", basefmt="-b")
print('Rotated', fourier_result.real[1:])

# Use less coefficients 
xi, yi = resample_2d(points, N)

contour_array = np.stack((np.roll(xi, -1), np.roll(yi, -1)), axis=1)
contour_complex = np.empty(contour_array.shape[:-1], dtype=complex)
contour_complex.real = contour_array[:, 0]
contour_complex.imag = contour_array[:, 1]
fourier_result = np.fft.fft(contour_complex)

truncated_fourier_result = np.copy(fourier_result)
truncated_fourier_result[30:] = 0
inverse_fourier_result = np.fft.ifft(truncated_fourier_result)
contour_reconstruct = np.array(
        [inverse_fourier_result.real, inverse_fourier_result.imag])
contour_reconstruct = np.transpose(contour_reconstruct)

# xi = np.roll(xi, -1)
# yi = np.roll(yi, -1)
xi = contour_reconstruct[:, 0]
yi = contour_reconstruct[:, 0]
ax[0, 4].scatter(xi[2:], yi[2:], c='blue')
ax[0, 4].scatter(xi[0], yi[0], c='red', label='Start')
ax[0, 4].scatter(xi[1], yi[1], c='Green', label='Second')
ax[0, 4].legend()
ax[0, 4].set_title('Use less coefficients')


# ax[1, 4].stem(list(range(len(fourier_result[1:]))), abs(fourier_result[1:]), 'b', markerfmt=" ", basefmt="-b")
# ax[1, 4].plot(fourier_result.real[1:], label='Real')
# ax[1, 4].plot(fourier_result.imag[1:], label='Imag')
# ax[1, 4].legend(loc='lower left')

# phase = np.arctan2(fourier_result.imag, fourier_result.real)
# ax[2, 4].stem(list(range(len(phase[1:]))), phase[1:], 'b', markerfmt=" ", basefmt="-b")
# print('Start point', fourier_result.real[1:])




# Normalize to centroid
xi, yi = resample_2d(points, N)
xc = np.mean(xi)
yc = np.mean(yi)

xi = xi-xc
yi = yi-yc

contour_array = np.stack((xi, yi), axis=1)
contour_complex = np.empty(contour_array.shape[:-1], dtype=complex)
contour_complex.real = contour_array[:, 0]
contour_complex.imag = contour_array[:, 1]
fourier_result = np.fft.fft(contour_complex)

ax[0, 5].scatter(xi[2:], yi[2:], c='blue')
ax[0, 5].scatter(xi[0], yi[0], c='red', label='Start')
ax[0, 5].scatter(xi[1], yi[1], c='Green', label='Second')
ax[0, 5].legend()
ax[0, 5].set_title('Norm. Centroid')

ax[1, 5].stem(list(range(len(fourier_result[1:]))), abs(fourier_result[1:]), 'b', markerfmt=" ", basefmt="-b")
# ax[1, 5].plot(fourier_result.real[1:], label='Real')
# ax[1, 5].plot(fourier_result.imag[1:], label='Imag')
# ax[1, 5].legend(loc='lower left')

phase = np.arctan2(fourier_result.imag, fourier_result.real)
ax[2, 5].stem(list(range(len(phase[1:]))), phase[1:], 'b', markerfmt=" ", basefmt="-b")
plt.show()
