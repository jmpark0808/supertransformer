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

all_ys = []
all_xs = []

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
    
    # plt.imshow(mark_boundaries(img, segments-1))
    # plt.show()
    regions = regionprops_table(segments, img, properties=('label', 'centroid'))
    
    if regions['label'][0] == 1:
        all_ys.append(regions['centroid-0'][0])
        all_xs.append(regions['centroid-1'][0])
    
xedges = np.linspace(-1, 1, 50)
yedges = np.linspace(-1, 1, 50)
all_ys = np.array(all_ys) - np.mean(all_ys)
all_xs = np.array(all_xs) - np.mean(all_xs)
np.save('centroid_shifts_0.npy', np.stack((all_xs, all_ys), axis=0))

assert(0)

all_centroids = np.load('./Analysis/centroid_shifts.npy')
distances = np.linalg.norm(all_centroids, axis=0)
all_xs = all_centroids[0]
all_ys = all_centroids[1]
densObj = kde( all_centroids )


def makeColours( vals ):
    colours = np.zeros( (len(vals),3) )
    norm = Normalize( vmin=vals.min(), vmax=vals.max() )

    #Can put any colormap you like here.
    colours = [cm.ScalarMappable( norm=norm, cmap='jet').to_rgba( val ) for val in vals]

    return colours

colours = makeColours( densObj.evaluate( all_centroids ) )
fig = plt.figure(figsize=(15, 10))
plt.scatter( all_centroids[0], all_centroids[1], c=densObj.evaluate( all_centroids ), cmap='jet' )
plt.colorbar()
plt.title('Kernel Density Estimation of Superpixel Centroids', fontsize=20)
plt.xlim(-5, 5)
plt.ylim(-5, 5)
plt.tight_layout()
plt.savefig('kde_centroids.pdf',format='pdf')

plt.show()

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






            