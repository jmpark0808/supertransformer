import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image

rectangle = np.zeros([100, 100])
rectangle[30:70, 40:60] = 1
rectangle = (rectangle*255).astype(np.uint8)





# plt.imshow(rectangle, cmap='gray')
# plt.show()

def euc_distance(pt1, pt2):
    y_diff = np.abs(pt2[1]-pt1[1])
    x_diff = np.abs(pt2[0]-pt1[0])
    return np.sqrt(y_diff**2+x_diff**2)


contour, hierarchy = cv2.findContours(rectangle, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
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

xi, yi = resample_2d(points, 14)
contour_array = np.stack((xi, yi), axis=1)
# plt.scatter(points[1:, 0], points[1:, 1], c='blue')
# plt.scatter(points[0, 0], points[0, 1], c='red')
# plt.show()

contour_complex = np.empty(contour_array.shape[:-1], dtype=complex)
contour_complex.real = contour_array[:, 0]
contour_complex.imag = contour_array[:, 1]
fourier_result = np.fft.fft(contour_complex)

# plt.scatter(xi[1:], yi[1:], c='blue')
# plt.scatter(xi[0], yi[0], c='red')
# plt.show()


# TRANSLATION
contour_array = np.stack((xi, yi), axis=1)+50
contour_complex = np.empty(contour_array.shape[:-1], dtype=complex)
contour_complex.real = contour_array[:, 0]
contour_complex.imag = contour_array[:, 1]
fourier_result = np.fft.fft(contour_complex)

# ROTATION

im = Image.fromarray(rectangle)
rotated = im.rotate(30)
rectangle_rotated = np.array(rotated)

contour, hierarchy = cv2.findContours(rectangle_rotated, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
points = contour[0][:, 0, :]

xi, yi = resample_2d(points, 14)
contour_array = np.stack((xi, yi), axis=1)
contour_complex = np.empty(contour_array.shape[:-1], dtype=complex)
contour_complex.real = contour_array[:, 0]
contour_complex.imag = contour_array[:, 1]
fourier_result = np.fft.fft(contour_complex)



# START POINT
print(xi)
print(np.roll(xi, -1))

contour_array = np.stack((np.roll(xi, -1), np.roll(yi, -1)), axis=1)
contour_complex = np.empty(contour_array.shape[:-1], dtype=complex)
contour_complex.real = contour_array[:, 0]
contour_complex.imag = contour_array[:, 1]
fourier_result = np.fft.fft(contour_complex)

print(fourier_result)
