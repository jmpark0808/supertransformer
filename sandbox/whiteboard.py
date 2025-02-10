import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
from skimage.draw import ellipse
from scipy.ndimage import rotate
import cv2

def compute_central_moments(binary_image):
    """
    Computes central moments of a binary image using skimage.measure.regionprops.
    
    Parameters:
    - binary_image: (2D numpy array) Binary image.

    Returns:
    - central_moments (numpy array): The computed central moments.
    """
    # label_image = measure.label(binary_image)  # Label connected components
    # props = measure.regionprops(label_image)
    # moments = props[0].moments_central 
    moments = measure.moments_central(binary_image)
    return moments

def log_moments(moments):
    moments_ = np.sign(moments)*np.log(np.abs(moments)+1e-10)
    moments_ = np.array([moments_[0,0], moments_[1, 1], moments_[2, 0],
                          moments_[0, 2], moments_[2, 1], moments_[1, 2], moments_[3, 0], moments_[0, 3]])
    return moments_ # Return the central moments of the largest region

def parse_moments(moments):
    moments_ = np.array([moments[0,0], moments[1, 1], moments[2, 0],
                          moments[0, 2], moments[2, 1], moments[1, 2], moments[3, 0], moments[0, 3]])
    return moments_ # Return the central moments of the largest region

def apply_translation(binary_image, shift_x, shift_y):
    """
    Translates a binary image by shifting pixels.

    Parameters:
    - binary_image: (2D numpy array) Binary image.
    - shift_x: (int) Shift along the x-axis.
    - shift_y: (int) Shift along the y-axis.

    Returns:
    - Translated binary image.
    """
    return np.roll(np.roll(binary_image, shift_y, axis=0), shift_x, axis=1)

def apply_horizontal_flip(binary_image):
    """
    Flips a binary image horizontally.
    
    Parameters:
    - binary_image: (2D numpy array) Binary image.

    Returns:
    - Flipped binary image.
    """
    return np.fliplr(binary_image)

def apply_rotation(binary_image, angle):
    """
    Rotates a binary image by a given angle.

    Parameters:
    - binary_image: (2D numpy array) Binary image.
    - angle: (float) Rotation angle in degrees.

    Returns:
    - Rotated binary image.
    """
    return rotate(binary_image, angle, reshape=False, mode='nearest')

def apply_scaling(binary_image, scale_factor):
    """
    Scales a binary image using OpenCV.

    Parameters:
    - binary_image: (2D numpy array) Binary image.
    - scale_factor: (float) Scaling factor.

    Returns:
    - Scaled binary image.
    """
    height, width = binary_image.shape
    new_size = (int(width * scale_factor), int(height * scale_factor))
    resized = cv2.resize(binary_image.astype(np.uint8), new_size, interpolation=cv2.INTER_NEAREST)

    # Pad or crop to maintain original size
    final_image = np.zeros_like(binary_image)
    min_h, min_w = min(final_image.shape[0], resized.shape[0]), min(final_image.shape[1], resized.shape[1])
    final_image[:min_h, :min_w] = resized[:min_h, :min_w]
    
    return final_image

# Create an asymmetrical "L" shape
image_size = (200, 200)
binary_image = np.zeros(image_size, dtype=np.uint8)

# Draw L-shape manually
binary_image[50:150, 50:80] = 1  # Vertical segment
binary_image[120:150, 50:130] = 1  # Horizontal segment

# Compute central moments for the original image
original_moments = compute_central_moments(binary_image)

# Apply transformations
translated_image = apply_translation(binary_image, shift_x=30, shift_y=20)
rotated_image = apply_rotation(binary_image, angle=45)
flipped_image = apply_horizontal_flip(binary_image)
scaled_image = apply_scaling(binary_image, scale_factor=1.5)

# Compute moments for transformed images
translated_moments = log_moments(compute_central_moments(translated_image))
rotated_moments = log_moments(compute_central_moments(rotated_image))
flipped_moments = log_moments(compute_central_moments(flipped_image))
scaled_moments = log_moments(compute_central_moments(scaled_image))

# for i in range(0, 360, 5):
#     rotated_image = apply_rotation(binary_image, angle=i)
#     rotated_moments = log_moments(compute_central_moments(rotated_image))
#     plt.bar(list(range(8)), rotated_moments.flatten())
#     plt.ylim(-20, 20)
#     plt.savefig(f'/home/eddie/Downloads/gif/{i}.png')
    
#     plt.clf()


# Display results
fig, axes = plt.subplots(1, 5, figsize=(15, 4))
axes[0].imshow(binary_image, cmap='gray'); axes[0].set_title("Original")
axes[1].imshow(translated_image, cmap='gray'); axes[1].set_title("Translated")
axes[2].imshow(rotated_image, cmap='gray'); axes[2].set_title("Rotated (45°)")
axes[3].imshow(flipped_image, cmap='gray'); axes[3].set_title("Horizontally Flipped")
axes[4].imshow(scaled_image, cmap='gray'); axes[4].set_title("Scaled (1.5x)")

plt.show()

# Print comparison of moments
print("\nCentral Moments Comparison:")
print(f"Original Moments:\n{original_moments}")
# print(f"Translated Moments (Should be same as original):\n{compute_central_moments(translated_image)}")
# print(f"Rotated Moments (Should change):\n{compute_central_moments(rotated_image)}")
# print(f"Flipped Moments (Should change in x-axis moments):\n{compute_central_moments(flipped_image)}")
# print(f"Scaled Moments (Should change significantly):\n{compute_central_moments(scaled_image)}")

fig, ax = plt.subplots(2, 5, figsize=(15,4))
columns = 8
ax[0,0].bar(list(range(columns)), log_moments(original_moments).flatten())
ax[0,1].bar(list(range(columns)), translated_moments.flatten())
ax[0,2].bar(list(range(columns)), rotated_moments.flatten())
ax[0,3].bar(list(range(columns)), flipped_moments.flatten())
ax[0,4].bar(list(range(columns)), scaled_moments.flatten())


def rotate_central_moments(moments, rotation_angle_degrees):
    """
    Computes the rotated central moments (up to third order) for a shape
    when rotated about its centroid by a given angle.
    
    Parameters:
    -----------
    moments : dict
        Dictionary with the following keys:
          - 'mu20', 'mu02', 'mu11' for second-order central moments.
          - 'mu30', 'mu03', 'mu21', 'mu12' for third-order central moments.
    rotation_angle_degrees : float
        Rotation angle in degrees.
        
    Returns:
    --------
    rotated_moments : dict
        Dictionary containing the rotated moments with the same keys.
    """
    # Convert degrees to radians
    theta = np.deg2rad(rotation_angle_degrees)
    
    # Extract second-order moments
    mu20 = moments[2, 0]
    mu02 = moments[0, 2]
    mu11 = moments[1, 1]
    
    # Rotate second-order moments
    mu20_rot = mu20 * (np.cos(theta)**2) + mu02 * (np.sin(theta)**2) - 2 * mu11 * np.sin(theta) * np.cos(theta)
    mu02_rot = mu20 * (np.sin(theta)**2) + mu02 * (np.cos(theta)**2) + 2 * mu11 * np.sin(theta) * np.cos(theta)
    mu11_rot = (mu20 - mu02) * np.sin(theta) * np.cos(theta) + mu11 * (np.cos(theta)**2 - np.sin(theta)**2)
    
    # Extract third-order moments
    mu30 = moments[3, 0]
    mu03 = moments[0, 3]
    mu21 = moments[2, 1]
    mu12 = moments[1, 2]
    
    # Rotate third-order moments
    mu30_rot = (mu30 * np.cos(theta)**3 
                - 3 * mu21 * np.cos(theta)**2 * np.sin(theta)
                + 3 * mu12 * np.cos(theta) * np.sin(theta)**2 
                - mu03 * np.sin(theta)**3)
    
    mu03_rot = (mu30 * np.sin(theta)**3 
                + 3 * mu21 * np.cos(theta) * np.sin(theta)**2
                + 3 * mu12 * np.cos(theta)**2 * np.sin(theta) 
                + mu03 * np.cos(theta)**3)
    
    mu21_rot = (mu30 * np.cos(theta)**2 * np.sin(theta)
                + mu21 * (np.cos(theta)**3 - 2*np.cos(theta)*np.sin(theta)**2)
                + mu12 * (np.sin(theta)**3 - 2*np.cos(theta)**2*np.sin(theta))
                + mu03 * np.cos(theta)*np.sin(theta)**2)
    
    mu12_rot = (mu30 * np.cos(theta) * np.sin(theta)**2
                + mu21 * (2*np.cos(theta)**2*np.sin(theta) - np.sin(theta)**3)
                + mu12 * (np.cos(theta)**3 - 2*np.cos(theta)*np.sin(theta)**2)
                - mu03 * np.cos(theta)**2 * np.sin(theta))
    
    
    rotated_moments = np.copy(moments)
    rotated_moments[2, 0] = mu20_rot
    rotated_moments[0, 2] = mu02_rot
    rotated_moments[1, 1] = mu11_rot

    rotated_moments[3, 0] = mu30_rot
    rotated_moments[0, 3] = mu03_rot
    rotated_moments[2, 1] = mu21_rot
    rotated_moments[1, 2] = mu12_rot
    
    return rotated_moments



# Transform moments
rotated_moments = log_moments(rotate_central_moments(original_moments, rotation_angle_degrees=45))
# scaled_moments = log_moments(transform_central_moments(original_moments, scale_factor=2))
# flipped_moments = log_moments(transform_central_moments(original_moments, horizontal_flip=True))
ax[1,2].bar(list(range(columns)), rotated_moments.flatten())
# ax[1,3].bar(list(range(columns)), flipped_moments.flatten())
# ax[1,4].bar(list(range(columns)), scaled_moments.flatten())
custom_ylim = (-40, 40)



# Setting the values for all axes.
plt.setp(ax, ylim=custom_ylim)

plt.show()
# Print results
# print("Original Moments:", original_moments)
# print("Rotated Moments (45°):", rotated_moments)
# print("Scaled Moments (2x):", scaled_moments)
# print("Flipped Moments:", flipped_moments)
print("Original Moments:", original_moments)
s1 = []
s2 = []

for i in range(0, 360, 5):
    # rotated_image = apply_rotation(binary_image, angle=i)
    # rotated_moments = log_moments(compute_central_moments(rotated_image))
    # fig, ax = plt.subplots(1, 3)
    # ax[0].bar(list(range(8)), rotated_moments.flatten())
    # rotated_moments = log_moments(rotate_central_moments(original_moments, rotation_angle_degrees=i))
    # ax[1].bar(list(range(8)), rotated_moments.flatten())
    # ax[2].bar(list(range(8)), log_moments(original_moments).flatten())
    # ax[0].set_ylim(-20, 20)
    # ax[1].set_ylim(-20, 20)
    # ax[2].set_ylim(-20, 20)
    # fig.savefig(f'/home/eddie/Downloads/gif/{i}.png')
    
    # plt.clf()
    rotated_image = apply_rotation(binary_image, angle=i)
    rotated_moments = parse_moments(compute_central_moments(rotated_image))

    t = parse_moments(rotate_central_moments(original_moments, rotation_angle_degrees=i))

    s1.append(rotated_moments)
    s2.append(t)
   
s1 = np.array(s1)
s2 = np.array(s2)

fig, ax = plt.subplots(1, 8)
for i in range(8):
    ax[i].plot(s1[:, i], label='Rotated')
    ax[i].plot(s2[:, i], label='Transformed')
    ax[i].legend()

plt.show()




