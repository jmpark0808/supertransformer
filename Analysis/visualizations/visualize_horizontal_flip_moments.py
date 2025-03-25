import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from skimage.measure import moments_central
plt.rcParams['text.usetex'] = True
rectangle = np.load('/home/eddie/waterloo/supertransformer/Analysis/sample_sp.npy')
rectangle_hf = np.fliplr(rectangle)
fig, ax = plt.subplots(1, 3, figsize=(12, 5))


# ax1 = plt.subplot2grid(shape=(3,4), loc=(0,1), colspan=2)
# ax2 = plt.subplot2grid((3,4), (1,0), colspan=2)
# ax3 = plt.subplot2grid((3,4), (1,2), colspan=2)
# ax4 = plt.subplot2grid((3,4), (2,0), colspan=2)
# ax5 = plt.subplot2grid((3,4), (2,2), colspan=2)
ax[0].imshow(rectangle, cmap='gray')
ax[0].set_title('Original', fontsize=15)
ax[1].imshow(rectangle_hf, cmap='gray')
ax[1].set_title('Horizontal Flipped', fontsize=15)
ax[0].axis('off')
ax[1].axis('off')
hf_moments = moments_central(rectangle_hf)
hf_moments = np.array([hf_moments[0,0], hf_moments[2,0], hf_moments[0,2], hf_moments[1,1],
               hf_moments[3,0], hf_moments[0,3], hf_moments[2,1], hf_moments[1,2]])
hf_moments = np.sign(hf_moments)*np.log(np.abs(hf_moments))

moments = moments_central(rectangle)
moments = np.array([moments[0,0], moments[2,0], moments[0,2], moments[1,1],
               moments[3,0], moments[0,3], moments[2,1], moments[1,2]])

moments_transformed = moments*np.array([1, 1, 1, -1, 1, -1, -1, 1])
moments_transformed = np.sign(moments_transformed)*np.log(np.abs(moments_transformed))
moments = np.sign(moments)*np.log(np.abs(moments))

x = np.arange(len(hf_moments))*1.5
ax[2].bar(x, hf_moments, width=0.3, facecolor='b', alpha=.5, label='Horizontal Flipped')
ax[2].bar(x+0.3, moments, width=0.3, facecolor='r', alpha=.5, label='Original')
ax[2].bar(x+0.6, moments_transformed, width=0.3, facecolor='g',  alpha=.5, label='Original Transformed')

ax[2].legend()
ax[2].set_xticks(x+0.3)
ax[2].set_xticklabels([r'$\mu_{00}$', r'$\mu_{02}$', r'$\mu_{20}$', r'$\mu_{11}$',
                      r'$\mu_{03}$', r'$\mu_{30}$', r'$\mu_{12}$', r'$\mu_{21}$'], fontsize=15)
ax[2].set_ylabel('Central Moments', fontsize=15)
ax[2].set_title('Central Moments Bar Plot', fontsize=15)
plt.tight_layout()
plt.savefig('/mnt/hdd/Figures/SuperFormer/horizontal_flip_moments.pdf', format='pdf')
plt.show()
# ax4.set_title('Downsample Reconstruction', fontsize=15)