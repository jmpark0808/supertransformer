import numpy as np
from PIL import Image
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
from skimage import data, io, segmentation, color
from skimage import graph
from skimage.morphology import disk
from skimage.segmentation import watershed
from skimage import data
from skimage.filters import rank
from skimage.util import img_as_ubyte
from skimage.segmentation import mark_boundaries, slic
from skimage.measure import regionprops_table
import os
from fast_slic.avx2 import SlicAvx2
from matplotlib.lines import Line2D
import torch

def scattered_partition(x, window_size, kernel_size, unfold1, unfold2):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    
    B, C, H, W = x.shape
    x = unfold1(x) # B C*k*k n
    x = x.reshape(B, C, kernel_size, kernel_size, -1) # B, C, k, k, n
    x = x.permute(0, 4, 1, 2, 3).contiguous().view(-1, C, kernel_size, kernel_size)# B*n, C, k, k 
    B_, _, _, _ = x.shape
    x = unfold2(x) # B*n, C*4*4, m
    x = x.reshape(B_, C, window_size, window_size, -1) # B*n, C, 4, 4, m
    x = x.permute(0, 4, 2, 3, 1).contiguous().view(-1, window_size*window_size, C) # B*n*m, 4, 4, C
    return x

image_path = '/mnt/dragon/Datasets/DUTS/DUTS-TR/Image/'
for i in os.listdir(image_path):
    img = Image.open(os.path.join(image_path, i)).convert('RGB')
    img = img.resize((320, 320))
    img = np.array(img)

    
    # slic = SlicAvx2(num_components=625, compactness=10)
    # segments = slic.iterate(img, max_iter=0)+1

    segments = slic(img, n_segments=1024,
                compactness=10,
                max_num_iter=10,
                convert2lab=True,
                enforce_connectivity=False,
                slic_zero=False)
    
    kernels = [4, 8, 16, 32]
    indices = torch.arange(0, 1024).reshape(1, 1, 32, 32).float()

    for k in kernels:

        unfold1 = torch.nn.Unfold(k, 1, stride=k)
        unfold2 = torch.nn.Unfold(4, k//4)
        ind = scattered_partition(indices, 4, k, unfold1, unfold2)
        for j, window in enumerate(ind):
            window_indices = torch.squeeze(window)
 
            

            out = color.label2rgb(segments, img, kind='avg', bg_label=0)
            out = segmentation.mark_boundaries(out, segments, (0, 0, 0))
            for sp in window_indices:

                ind = np.argwhere(segmentation.find_boundaries(segments == (int(sp)+1), mode='inner'))
  
                

                out[ind[:, 0], ind[:, 1]] = [255, 0, 0]
                
    
            
            # plt.imshow(out)
            
            # plt.show()
            plt.imshow(out)
            plt.savefig(f'/home/eddie/Downloads/gif_swin/{k}/{j}')
        
    

    assert(0)

