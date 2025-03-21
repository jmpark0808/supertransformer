import torch
import torch.nn.functional as F
import numpy as np
from skimage.feature import local_binary_pattern

def compute_superpixel_representation_batch(images, lbp_images, segmentation_maps, num_superpixels):
    """
    Computes the superpixel representation for a batch of images.
    - images: (B, H, W, 3)  -> RGB images
    - lbp_images: (B, H, W) -> Local Binary Pattern images
    - segmentation_maps: (B, H, W) -> Superpixel labels
    - num_superpixels: Scalar (assumes all images have the same number)
    
    Returns:
    - rgb_mean: (B, num_superpixels, 3)
    - rgb_std: (B, num_superpixels, 3)
    - lbp_hist: (B, num_superpixels, num_bins)
    - shape_moments: (B, num_superpixels, 7)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Move tensors to GPU
    images = torch.tensor(images, dtype=torch.float32, device=device)  # (B, H, W, 3)
    lbp_images = torch.tensor(lbp_images, dtype=torch.int64, device=device)  # (B, H, W)
    segmentation_maps = torch.tensor(segmentation_maps, dtype=torch.int64, device=device)  # (B, H, W)

    B, H, W, C = images.shape  # Batch size, height, width, channels
    
    # Flatten spatial dimensions
    images_flat = images.view(B, -1, C)  # (B, H*W, 3)
    lbp_flat = lbp_images.view(B, -1)  # (B, H*W)
    seg_flat = segmentation_maps.view(B, -1)  # (B, H*W)

    ### 1. Compute Mean and Standard Deviation of RGB per Superpixel
    ones = torch.ones_like(seg_flat, dtype=torch.float32, device=device)
    superpixel_sizes = torch.zeros((B, num_superpixels), device=device).scatter_reduce_(1, seg_flat, ones, reduce="sum")

    rgb_sum = torch.zeros((B, num_superpixels, C), device=device).scatter_reduce_(1, seg_flat.unsqueeze(-1).expand(-1, -1, C), images_flat, reduce="sum")
    rgb_mean = rgb_sum / (superpixel_sizes.unsqueeze(-1) + 1e-6) # (B, num_superpixels, 3)

    rgb_sq_sum = torch.zeros((B, num_superpixels, C), device=device).scatter_reduce_(1, seg_flat.unsqueeze(-1).expand(-1, -1, C), images_flat ** 2, reduce="sum")
    rgb_std = torch.sqrt(rgb_sq_sum / (superpixel_sizes.unsqueeze(-1) + 1e-6) - rgb_mean ** 2)

    ### 2. Compute LBP Histogram per Superpixel
    num_bins = lbp_flat.max().item() + 1  # Determine LBP histogram size dynamically
    lbp_hist = torch.zeros((B, num_superpixels, num_bins), device=device).scatter_add_(1, seg_flat.unsqueeze(-1).expand(-1, -1, num_bins), F.one_hot(lbp_flat, num_classes=num_bins).float())

    ### 3. Compute Shape Central Moments (7 Features) per Superpixel
    y_coords, x_coords = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
    x_coords, y_coords = x_coords.flatten(), y_coords.flatten()  # Flatten for indexing
    
    x_sum = torch.zeros((B, num_superpixels), device=device).scatter_reduce_(1, seg_flat, x_coords.float().unsqueeze(0).expand(B, -1), reduce="sum")
    y_sum = torch.zeros((B, num_superpixels), device=device).scatter_reduce_(1, seg_flat, y_coords.float().unsqueeze(0).expand(B, -1), reduce="sum")
    centroid_x = x_sum / (superpixel_sizes + 1e-6)
    centroid_y = y_sum / (superpixel_sizes + 1e-6)

    # Compute Central Moments (up to order 3)
    dx = x_coords.unsqueeze(0) - centroid_x.gather(1, seg_flat)  # (B, H*W)
    dy = y_coords.unsqueeze(0) - centroid_y.gather(1, seg_flat)  # (B, H*W)

    
    mu20 = torch.zeros((B, num_superpixels), device=device).scatter_reduce_(1, seg_flat, dx ** 2, reduce="sum") 
    mu02 = torch.zeros((B, num_superpixels), device=device).scatter_reduce_(1, seg_flat, dy ** 2, reduce="sum") 
    mu11 = torch.zeros((B, num_superpixels), device=device).scatter_reduce_(1, seg_flat, dx * dy, reduce="sum") 

    mu30 = torch.zeros((B, num_superpixels), device=device).scatter_reduce_(1, seg_flat, dx ** 3, reduce="sum") 
    mu03 = torch.zeros((B, num_superpixels), device=device).scatter_reduce_(1, seg_flat, dy ** 3, reduce="sum")
    mu21 = torch.zeros((B, num_superpixels), device=device).scatter_reduce_(1, seg_flat, dx ** 2 * dy, reduce="sum") 
    mu12 = torch.zeros((B, num_superpixels), device=device).scatter_reduce_(1, seg_flat, dx * dy ** 2, reduce="sum") 

    shape_moments = torch.stack([superpixel_sizes, mu11, mu20, mu02, mu21, mu12, mu30, mu03], dim=2)  # (B, num_superpixels, 7)
    
    return torch.cat((centroid_y, centroid_x, rgb_mean, rgb_std, shape_moments, lbp_hist), dim=2)



from PIL import Image
from skimage.segmentation import slic
from skimage.measure import regionprops_table
from skimage.measure import moments_central
import torchvision.transforms as transforms
import matplotlib.pylab as plt
img = Image.open('/home/eddie/Datasets/DUTS/DUTS-TE/Image/ILSVRC2012_test_00000003.jpg').convert('RGB')
mask = Image.open('/home/eddie/Datasets/DUTS/DUTS-TE/Mask/ILSVRC2012_test_00000003.png').convert('L')

img_np = np.array(img.resize((224, 224)))
img_gray = np.array(img.convert('L').resize((224, 224)))

def lbp(region, intensities):
    (hist, _) = np.histogram(intensities[region].ravel(),
            bins=np.arange(0, 8+3),
            range=(0, 8+2))
    hist = hist.astype("float")
    # hist /= (hist.sum() + 1e-7)
    return hist

segments = slic(img_np, n_segments=3136,
            compactness=10,
            max_num_iter=10,
            convert2lab=True,
            enforce_connectivity=False,
            slic_zero=False)

       
def image_stdev(region, intensities):
    # note the ddof arg to get the sample var if you so desire!
    return np.std(intensities[region])
        

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
    moments_ = moments_central(binary_image)
    # moments_ = np.sign(moments)*np.log(np.abs(moments)+1e-10)
    moments_ = np.array([moments_[0,0], moments_[1, 1], moments_[2, 0],
                          moments_[0, 2], moments_[2, 1], moments_[1, 2], moments_[3, 0], moments_[0, 3]])
    return moments_

def fourier_descriptors(region):
    moments = compute_central_moments(region)
    
    # return np.array(amp)
    return moments  
                
lbp_np = local_binary_pattern(img_gray, 8, 1, method='uniform')
regions_lbp = regionprops_table(segments, intensity_image=lbp_np, extra_properties=[lbp])

regions = regionprops_table(segments, intensity_image=img_np, properties=('label', 'centroid', 'intensity_mean',
                                                                            'coords'), extra_properties=[image_stdev, fourier_descriptors])#, polarize])
num_seg = 3136
seq_len = len(regions['label'])
seq_mask = np.zeros([num_seg])
label = regions['label']
# features = np.zeros([self.num_seg, 8+(self.resample_points-1)*2+10])


features = np.zeros([num_seg, 8+8+10])

# for i in range((self.resample_points-1)*2):
for i in range(8):
    features[label-1, 8+i] = regions[f'fourier_descriptors-{i}']


features[label-1, 0] = regions['centroid-0']
features[label-1, 1] = regions['centroid-1']

features[label-1, 2] = regions['intensity_mean-0']
features[label-1, 3] = regions['intensity_mean-1']
features[label-1, 4] = regions['intensity_mean-2']
features[label-1, 5] = regions['image_stdev-0']
features[label-1, 6] = regions['image_stdev-1']
features[label-1, 7] = regions['image_stdev-2']

for ind in range(8+2):
    # features[label-1, ind+8+(self.resample_points-1)*2] = regions_lbp[f'lbp-{ind}']
    features[label-1, ind+8+8] = regions_lbp[f'lbp-{ind}']

features_np = np.copy(features)



mean, std, lbp_features, moments = compute_superpixel_representation_batch(np.stack((img_np, img_np), 0), np.stack((lbp_np, lbp_np), 0), np.stack(((segments-1, segments-1)), 0), 3136)
features_tensor = torch.cat((mean, std,  moments, lbp_features), dim=2)

print(np.sum(features_np[:, 2:]-features_tensor.detach().cpu().numpy().squeeze()[0, :, :]))
print(features_np[0, 2:])
print(features_tensor.detach().cpu().numpy().squeeze()[0, 0, :])