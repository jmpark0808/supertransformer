

import matplotlib.pyplot as plt
from PIL import Image
import os
import numpy as np
from skimage.segmentation import slic, mark_boundaries
from skimage.measure import regionprops_table

image_1 = '/home/eddie/Datasets/DUTS/DUTS-TE/Image/ILSVRC2012_test_00000003.jpg'
mask_1 = '/home/eddie/Datasets/DUTS/DUTS-TE/Mask/ILSVRC2012_test_00000003.png'

image_2 = '/home/eddie/Datasets/DUTS/DUTS-TE/Image/ILSVRC2012_test_00000439.jpg'
mask_2 = '/home/eddie/Datasets/DUTS/DUTS-TE/Mask/ILSVRC2012_test_00000439.png'



img_1 = Image.open(image_1)
img_2 = Image.open(image_2)


mask_1 = Image.open(mask_1)
mask_2 = Image.open(mask_2)
img_1 = img_1.convert('RGB').resize((224, 224))
img_2 = img_2.convert('RGB').resize((224, 224))
mask_1 = mask_1.convert('L').resize((224, 224))
mask_2 = mask_2.convert('L').resize((224, 224))

img_1 = np.array(img_1)
img_2 = np.array(img_2)

mask_1 = np.array(mask_1)
mask_2 = np.array(mask_2)

num_seg = 3136

segments_1 = slic(img_1, n_segments=num_seg,
compactness=10,
max_num_iter=10,
convert2lab=True,
enforce_connectivity=False,
slic_zero=False)

segments_2 = slic(img_2, n_segments=num_seg,
compactness=10,
max_num_iter=10,
convert2lab=True,
enforce_connectivity=False,
slic_zero=False)



indices = np.arange(0, num_seg).reshape((int(num_seg**0.5), int(num_seg**0.5)))
cut_indices = indices[int(num_seg**0.5//2):, int(num_seg**0.5//2):]

regions_1 = regionprops_table(segments_1, img_1, properties=('label', 'centroid', 'area', 'intensity_mean',
                                                                        'coords',))

regions_2 = regionprops_table(segments_2, img_2, properties=('label', 'centroid', 'area', 'intensity_mean',
                                                                        'coords',))


seq_mask_1 = np.zeros([num_seg])
seq_mask_2 = np.zeros([num_seg])
# seq_mask_2 = np.zeros([max(regions_2['label'])])


seq_image_1 = np.zeros([num_seg, 3])
seq_image_2 = np.zeros([num_seg, 3])
# seq_image_2 = np.zeros([max(regions_2['label']), 3])
# assert len(regions['label']) == max(regions['label']), 'Wrong number of labels'

for ind, coord, r, g, b in zip(regions_1['label'], regions_1['coords'], regions_1['intensity_mean-0'], regions_1['intensity_mean-1'], regions_1['intensity_mean-2']):
    # seq_mask_1[ind-1] = 1 if np.sum(mask_1[coord[:, 0], coord[:, 1]])/len(coord[:, 0]) >= 0.5 else 0
    seq_mask_1[ind-1] = np.sum(mask_1[coord[:, 0], coord[:, 1]])/len(coord[:, 0])
    seq_image_1[ind-1] = [r, g, b]

for ind, coord, r, g, b in zip(regions_2['label'], regions_2['coords'], regions_2['intensity_mean-0'], regions_2['intensity_mean-1'], regions_2['intensity_mean-2']):
    seq_mask_2[ind-1] = np.sum(mask_2[coord[:, 0], coord[:, 1]])/len(coord[:, 0])
    seq_image_2[ind-1] = [r, g, b]
    
mask_cutmix = np.zeros_like((224, 224))
seq_cutmix = np.zeros_like((224, 224, 3))

mask_cutmix = seq_mask_1[segments_1-1].reshape([img_1.shape[0], img_1.shape[1]])
seq_cutmix = seq_image_1[segments_1-1].reshape([img_1.shape[0], img_1.shape[1], 3])

for row_ind, row in enumerate(segments_2):
    for col_ind, col in enumerate(row):
        if col-1 in cut_indices:
            mask_cutmix[row_ind, col_ind] = seq_mask_2[col-1]
            seq_cutmix[row_ind, col_ind, :] = seq_image_2[col-1, :]

mask_1 = seq_mask_1[segments_1-1].reshape([img_1.shape[0], img_1.shape[1]])
mask_2 = seq_mask_2[segments_2-1].reshape([img_2.shape[0], img_2.shape[1]])

image_1 = seq_image_1[segments_1-1].reshape([img_1.shape[0], img_1.shape[1], 3])/255.
image_2 = seq_image_2[segments_2-1].reshape([img_2.shape[0], img_2.shape[1], 3])/255.
# for seg_1, seg_2 in zip(segments_1, segments_2):
#     if seg_1 not in cut_indices:
#         seq_mask_1[seg_1-1]
#     if seg_2 in cut_indices:


# mask_image_2 = seq_mask_2[segments_2-1].reshape([img_2.shape[0], img_2.shape[1]])

seq_cutmix = seq_cutmix/255.
# seq_image_2 = seq_image_2[segments_2-1].reshape([img_2.shape[0], img_2.shape[1], 3])/255.

fig, ax = plt.subplots(2, 3, figsize=(9, 6))
ax[0,0].set_title('Superpixel Set A', fontsize=10)
ax[0,0].axis('off')
ax[0, 0].imshow(image_1)
ax[0,1].set_title('Superpixel Set B', fontsize=10)
ax[0,1].axis('off')
ax[0, 1].imshow(image_2)
ax[0,2].set_title('Superpixel Set CutMix', fontsize=10)
ax[0,2].axis('off')
ax[0, 2].imshow(seq_cutmix)
ax[1,0].set_title('Superpixel Mask A', fontsize=10)
ax[1,0].axis('off')
ax[1, 0].imshow(mask_1, cmap='gray')
ax[1,1].set_title('Superpixel Mask B', fontsize=10)
ax[1,1].axis('off')
ax[1, 1].imshow(mask_2, cmap='gray')
ax[1,2].set_title('Superpixel Mask Cutmix', fontsize=10)
ax[1,2].axis('off')
ax[1, 2].imshow(mask_cutmix, cmap='gray')
fig.subplots_adjust(hspace=0.281, left=0, right=1, wspace=0, top=0.925, bottom=0)
# plt.subplot_tool()
plt.savefig('/mnt/hdd/Figures/SuperFormer/Cutmix.pdf', format='pdf')
plt.show()