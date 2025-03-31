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


num_images = 20

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
    

#     centroids_y = np.zeros([3136])
#     centroids_x = np.zeros([3136])

#     for label, x, y in zip(regions['label'],regions['centroid-1'], regions['centroid-0'] ):
#         centroids_y[label-1] = y
#         centroids_x[label-1] = x
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


# np.save('./Analysis/centroid_shifts.npy', np.stack((all_shifted_xs, all_shifted_ys), axis=0))

# assert(0)

all_centroids = np.load('./Analysis/centroid_shifts.npy')
all_centroids = all_centroids[:, :20]
all_centroids = all_centroids.reshape(2, -1)
non_zero_mask = np.logical_and(all_centroids[0] == 0, all_centroids[1] == 0)
all_centroids = all_centroids[:, ~non_zero_mask]
distances = np.linalg.norm(all_centroids, axis=0)
all_xs = all_centroids[0]
all_ys = all_centroids[1]
densObj = kde( all_centroids)

x_grid = np.linspace(-8, 8, 1000)
y_grid = np.linspace(-8, 8, 1000)
X, Y = np.meshgrid(x_grid, y_grid)
Z = densObj.evaluate(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
Z /= np.sum(Z)

Z_log = np.log(Z)
Z_sqrt = np.power(Z, 0.25)

# def makeColours( vals ):
#     colours = np.zeros( (len(vals),3) )
#     norm = Normalize( vmin=vals.min(), vmax=vals.max() )

#     #Can put any colormap you like here.
#     colours = [cm.ScalarMappable( norm=norm, cmap='jet').to_rgba( val ) for val in vals]

#     return colours

# colours = makeColours( densObj.evaluate( all_centroids ) )
fig, ax = plt.subplots(1, 2, figsize=(15, 5))

ax[0].scatter( all_centroids[0], all_centroids[1], c='b', alpha=0.5)
contour = ax[1].contourf(X, Y, Z_sqrt, levels=50, cmap='jet')
# contour = ax[2].contourf(X, Y, Z_log, levels=50, cmap='jet')

# cbar = fig.colorbar(contour, ax= ax[1])
# cbar.set_label('Probability Density', fontsize=15)

# cbar = plt.colorbar()
# 
ax[0].set_title('All Superpixel Centroids', fontsize=15)
ax[1].set_title('KDE of All Superpixel Centroids', fontsize=15)
# ax[2].set_title('KDE w/ log', fontsize=20)
ax[1].set_xlim(-8, 8)
ax[1].set_ylim(-8, 8)
# ax[2].set_xlim(-8, 8)
# ax[2].set_ylim(-8, 8)
ax[0].set_xlim(-8, 8)
ax[0].set_ylim(-8, 8)
ax[0].set_aspect('equal')
ax[1].set_aspect('equal')
# ax[2].set_aspect('equal')
fig.tight_layout()
fig.savefig('/mnt/hdd/Figures/SuperFormer/kde_centroids.pdf',format='pdf')

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






            