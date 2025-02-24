import numpy as np
from skimage.measure import regionprops, label
import matplotlib.pyplot as plt
import math
image = np.zeros((100, 100), dtype=np.uint8)
rr, cc = np.ogrid[0:100, 0:100]
ellipse = ((rr - 50) ** 2 / 30 ** 2 + (cc - 50) ** 2 / 20 ** 2) <= 1
image[ellipse] = 1

# Label the image and calculate region properties
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

# Calculate endpoints of major and minor axes
y0, x0 = region.centroid
x1 = x0 + np.cos(orientation) * 0.5 * minor_axis_length
y1 = y0 - np.sin(orientation) * 0.5 * minor_axis_length

x2 = x0 - np.sin(orientation) * 0.5 * major_axis_length
y2 = y0 - np.cos(orientation) * 0.5 * major_axis_length

# Plot the results
fig, ax = plt.subplots()
ax.imshow(image, cmap=plt.cm.gray)
ax.plot((x0, x1), (y0, y1), '-r', linewidth=2.5)
ax.plot((x0, x2), (y0, y2), '-b', linewidth=2.5)
ax.plot(x0, y0, '.g', markersize=15)

ax.set_title('Ellipse with Major and Minor Axes')
plt.show()

print(f"Major axis length: {major_axis_length:.2f}")
print(f"Minor axis length: {minor_axis_length:.2f}")