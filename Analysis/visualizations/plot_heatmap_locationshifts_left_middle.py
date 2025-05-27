import os
import matplotlib.pyplot as plt
import cv2
from skimage import io
from skimage.segmentation import mark_boundaries, slic, quickshift
from skimage.measure import regionprops_table
import numpy as np
from PIL import Image
from tqdm import tqdm
import pickle
import time
from fast_slic.avx2 import SlicAvx2
import torch
from torch_geometric.utils import scatter
import torch.nn.functional as F
from torch_scatter import scatter_std
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde as kde
from matplotlib.colors import Normalize
from matplotlib import cm
from matplotlib.patches import Rectangle

dataset_images = '/home/eddie/Datasets/DUTS/DUTS-TR/Image'
masks = '/home/eddie/Datasets/DUTS/DUTS-TR/Mask'


num_images = 3000

# all_ys = []
# all_xs = []

# for file in tqdm(os.listdir(dataset_images)[:num_images]):
#     name = file.split('.jpg')[0]
#     image = os.path.join(dataset_images, name+'.jpg')
#     mask = os.path.join(masks, name+'.png')

#     img = Image.open(image)
#     msk = Image.open(mask)
#     img = img.convert('RGB').resize((448, 448))
#     msk = msk.convert('L').resize((448, 448))
#     img = np.array(img)
#     msk = np.array(msk)
    
    
#     msk[msk>125] = 255
#     msk[msk<=125] = 0

#     segments = slic(img, n_segments=3136,
#     compactness=10,
#     max_num_iter=10,
#     convert2lab=True,
#     enforce_connectivity=False,
#     slic_zero=False)
    
#     # plt.imshow(mark_boundaries(img, segments-1))
#     # plt.show()
#     regions = regionprops_table(segments, img, properties=('label', 'centroid'))
    

#     centroids_y = np.zeros([2])
#     centroids_x = np.zeros([2])

#     for label, x, y in zip(regions['label'],regions['centroid-1'], regions['centroid-0'] ):
#         if label == 1:
#             centroids_y[0] = y
#             centroids_x[0] = x
#         elif label == 1596:
#             centroids_y[1] = y
#             centroids_x[1] = x

#     all_ys.append(centroids_y)
#     all_xs.append(centroids_x)
    

# all_xs = np.stack(all_xs, axis=0)
# all_ys = np.stack(all_ys, axis=0)

# mask = all_xs != 0
# sum_values_x = np.sum(all_xs*mask, axis=0)
# sum_values_y = np.sum(all_ys*mask, axis=0)
# count_nonzero = np.sum(mask, axis=0)

# mean_x = sum_values_x / count_nonzero
# mean_y = sum_values_y / count_nonzero

# all_shifted_xs = np.where(mask, all_xs - mean_x, all_xs)
# all_shifted_ys = np.where(mask, all_ys - mean_y, all_ys)


# np.save('./Analysis/centroid_shifts_left_middle.npy', np.stack((all_shifted_xs, all_shifted_ys), axis=0))

# assert(0)

all_centroids = np.load('./Analysis/centroid_shifts_left_middle.npy')

all_centroids_left = all_centroids[:, :, 0]
all_centroids_middle = all_centroids[:, :, 1]


non_zero_mask_left = np.logical_and(all_centroids_left[0] == 0, all_centroids_left[1] == 0)
non_zero_mask_middle = np.logical_and(all_centroids_middle[0] == 0, all_centroids_middle[1] == 0)

all_centroids_left = all_centroids_left[:, ~non_zero_mask_left]
all_centroids_middle = all_centroids_middle[:, ~non_zero_mask_middle]

# distances = np.linalg.norm(all_centroids, axis=0)
all_centroids_left[1] = -all_centroids_left[1]
all_centroids_middle[1] = -all_centroids_middle[1]
densObj_left = kde( all_centroids_left)
densObj_middle = kde( all_centroids_middle)

x_grid = np.linspace(-8, 8, 1000)
y_grid = np.linspace(-8, 8, 1000)
X, Y = np.meshgrid(x_grid, y_grid)
Z_left = densObj_left.evaluate(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
Z_middle = densObj_middle.evaluate(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)

Z_left /= np.sum(Z_left)
Z_middle /= np.sum(Z_middle)

Z_log_left = np.log(Z_left)
Z_sqrt_left = np.power(Z_left, 0.25)


Z_log_middle = np.log(Z_middle)
Z_sqrt_middle = np.power(Z_middle, 0.25)

# def makeColours( vals ):
#     colours = np.zeros( (len(vals),3) )
#     norm = Normalize( vmin=vals.min(), vmax=vals.max() )

#     #Can put any colormap you like here.
#     colours = [cm.ScalarMappable( norm=norm, cmap='jet').to_rgba( val ) for val in vals]

#     return colours

# colours = makeColours( densObj.evaluate( all_centroids ) )

import matplotlib as mpl
fig = plt.figure(figsize=(15, 10))
spec = mpl.gridspec.GridSpec(ncols=3, nrows=4, wspace=0.1,hspace=0.3,left=0.01,right=0.99,top=0.95,bottom=0.01)

ax1 = fig.add_subplot(spec[1:3,0])
ax2 = fig.add_subplot(spec[0:2,1])
ax3 = fig.add_subplot(spec[2:4,1])
ax4 = fig.add_subplot(spec[0:2,2])
ax5 = fig.add_subplot(spec[2:4,2])

cell_size = 16
x_start, x_end = 0, 448  # custom horizontal span
y_start, y_end = 0, 448  # custom vertical span

# Draw vertical lines
for x in range(x_start, x_end + 1, cell_size):
    ax1.plot([x, x], [y_start, y_end], color='red', linewidth=0.1, zorder=0)

# Draw horizontal lines
for y in range(y_start, y_end + 1, cell_size):
    ax1.plot([x_start, x_end], [y, y], color='red', linewidth=0.1, zorder=0)
# ax1.plot([0, 448, 448, 0, 0], [0, 0, 448, 448, 0], c='red', linewidth=5)
ax1.scatter([8], [440], marker='*', s=500, zorder=10, c='blue')
ax1.scatter([216], [232], marker='*', s=500, zorder=10, c='green')
ax1.text(20, 410, 'A', fontsize=20, zorder=10, fontweight='bold')
ax1.text(229, 204, 'B', fontsize=20, zorder=10, fontweight='bold')




ax2.scatter( all_centroids_left[0], all_centroids_left[1], c='b', alpha=0.5)
ax3.scatter( all_centroids_middle[0], all_centroids_middle[1], c='b', alpha=0.5)
contour0 = ax4.contourf(X, Y, Z_sqrt_left, levels=50, cmap='jet')
contour1 = ax5.contourf(X, Y, Z_sqrt_middle, levels=50, cmap='jet')
# contour0 = ax[0, 1].imshow(Z_left, extent=[-8, 8, -8, 8], cmap=
# 'jet')
# contour1 = ax[1, 1].imshow(Z_middle, extent=[-8, 8, -8, 8], cmap='jet')

# cbar = fig.colorbar(contour0, ax= ax[1])
# cbar.set_label('Probability Density', fontsize=15)

# cbar0 = plt.colorbar(contour0, ax= ax[0, 1])
# cbar1 = plt.colorbar(contour1, ax= ax[1, 1])
# 
ax2.set_title('"A" Superpixel Centroids', fontsize=15)
ax3.set_title('"B" Superpixel Centroids', fontsize=15)

ax4.set_title('KDE of "A" Superpixel Centroids', fontsize=15)
ax5.set_title('KDE of "B" Superpixel Centroids', fontsize=15)

ax1.set_xlim(-10, 458)
ax1.set_ylim(-10, 458)
ax2.set_xlim(-8, 8)
ax2.set_ylim(-8, 8)
ax3.set_xlim(-8, 8)
ax3.set_ylim(-8, 8)
ax4.set_xlim(-8, 8)
ax4.set_ylim(-8, 8)
ax5.set_xlim(-8, 8)
ax5.set_ylim(-8, 8)
ax1.axis('off')
ax2.set_aspect('equal')
ax3.set_aspect('equal')
ax4.set_aspect('equal')
ax5.set_aspect('equal')
# fig.tight_layout()
fig.savefig('/mnt/hdd/Figures/SuperFormer/kde_centroids_left_middle.pdf',format='pdf')

plt.show()
assert(0)

ind = 0
for file in tqdm(os.listdir(dataset_images)[:num_images]):
    name = file.split('.jpg')[0]
    image = os.path.join(dataset_images, name+'.jpg')
    mask = os.path.join(masks, name+'.png')

    img = Image.open(image)
    msk = Image.open(mask)
    img = img.convert('RGB').resize((448, 448))
    msk = msk.convert('L').resize((448, 448))
    img = np.array(img)
    msk = np.array(msk)
    
    
    msk[msk>125] = 255
    msk[msk<=125] = 0

    segments = slic(img, n_segments=3136,
    compactness=10,
    max_num_iter=10,
    convert2lab=True,
    enforce_connectivity=False,
    slic_zero=False)
    
    regions = regionprops_table(segments, img, properties=('label', 'centroid'))
    
    
    if regions['label'][1000] == 1001:
        if distances[ind] >= 3:
            segments = slic(img, n_segments=784,
                compactness=10,
                max_num_iter=10,
                convert2lab=True,
                enforce_connectivity=False,
                slic_zero=False)
            regions = regionprops_table(segments, img, properties=('label', 'centroid'))
            s_ind = 380
            plt.imshow(mark_boundaries(img, segments-1))
            plt.scatter(regions['centroid-1'][s_ind], regions['centroid-0'][s_ind], c='red', s=10)
            plt.gca().add_patch(Rectangle((regions['centroid-1'][s_ind]-30,regions['centroid-0'][s_ind]-30),60,60,linewidth=2,edgecolor='r',facecolor='none'))
            plt.axis('off')
            plt.show()
            print(regions['centroid-1'][s_ind], regions['centroid-1'][s_ind], mark_boundaries(img, segments-1).shape)
            plt.imshow(mark_boundaries(img, segments-1)[int(regions['centroid-0'][s_ind]-30):int(regions['centroid-0'][s_ind]+30),
                                                         int(regions['centroid-1'][s_ind]-30):int(regions['centroid-1'][s_ind]+30)])
            plt.scatter([30], [30], c='red', s=10)
            plt.text(31, 31, f'x: {int(regions["centroid-1"][s_ind])}, y: {int(regions["centroid-0"][s_ind])}')
            plt.axis('off')
            plt.show()


        ind += 1






            