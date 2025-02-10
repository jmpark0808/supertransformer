import numpy as np
import cv2
from scipy.ndimage import rotate
from skimage.measure import moments_central

def create_shape(size=100):
    shape = np.zeros((size, size), dtype=np.uint8)
    cv2.rectangle(shape, (25, 25), (75, 75), 255, -1)
    return shape

def calculate_moments(image):
    m = moments_central(image)
    return (m[0, 0], m[1, 0], m[0, 1], 
            m[1, 1], m[2, 0], m[0, 2],
            m[3, 0], m[2, 1], m[1, 2], m[0, 3])

def print_moments(moments):
    labels = ['m00', 'm10', 'm01', 
              'm11', 'm20', 'm02',
              'm30', 'm21', 'm12', 'm03']
    for label, moment in zip(labels, moments):
        print(f"{label}: {moment:.4f}", end=", ")
    print()

# Create original shape
original = create_shape()

# Calculate moments for original shape
original_moments = calculate_moments(original)

print("Original moments:")
print_moments(original_moments)

# Rotate shape
rotated = rotate(original, angle=45, reshape=False)

# Calculate moments for rotated shape
rotated_moments = calculate_moments(rotated)

print("\nRotated moments:")
print_moments(rotated_moments)

# Calculate relative changes
relative_changes = [(r - o) / o * 100 if o != 0 else float('inf') for r, o in zip(rotated_moments, original_moments)]

print("\nRelative changes (%):")
for label, change in zip(['m00', 'm10', 'm01', 'm11', 'm20', 'm02', 'm30', 'm21', 'm12', 'm03'], relative_changes):
    print(f"{label}: {change:.2f}%")
