import torch
import torch.nn.functional as F

def compute_superpixel_representation_batch(images, lbp_images, segmentation_maps, masks, num_superpixels):
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
    
    with torch.no_grad():

        # Move tensors to GPU

        B, C, H, W = images.shape  # Batch size, height, width, channels
        
        # Flatten spatial dimensions
        images_flat = images.permute(0, 2, 3, 1).view(B, -1, C)  # (B, H*W, 3)
        lbp_flat = lbp_images.view(B, -1).long()  # (B, H*W)
        seg_flat = segmentation_maps.view(B, -1)  # (B, H*W)
        masks_flat = masks.view(B, -1) # (B, H*W)

        ### 1. Compute Mean and Standard Deviation of RGB per Superpixel
        ones = torch.ones_like(seg_flat, dtype=torch.float32, device=device)
        superpixel_sizes = torch.zeros((B, num_superpixels), device=device).scatter_reduce_(1, seg_flat, ones, reduce="sum")

        rgb_sum = torch.zeros((B, num_superpixels, C), device=device).scatter_reduce_(1, seg_flat.unsqueeze(-1).expand(-1, -1, C), images_flat, reduce="sum")
        rgb_mean = rgb_sum / (superpixel_sizes.unsqueeze(-1) + 1e-6) # (B, num_superpixels, 3)

        rgb_sq_sum = torch.zeros((B, num_superpixels, C), device=device).scatter_reduce_(1, seg_flat.unsqueeze(-1).expand(-1, -1, C), images_flat ** 2, reduce="sum")
        rgb_std = torch.sqrt(torch.clamp(rgb_sq_sum / (superpixel_sizes.unsqueeze(-1) + 1e-6) - rgb_mean ** 2, min=0.0))
        
        ### 2. Compute LBP Histogram per Superpixel
        num_bins = 10 # Determine LBP histogram size dynamically
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

        seq_masks = torch.zeros((B, num_superpixels), device=device).scatter_reduce_(1, seg_flat, masks_flat, reduce="mean")

        shape_moments = torch.stack([superpixel_sizes, mu11, mu20, mu02, mu21, mu12, mu30, mu03], dim=2)  # (B, num_superpixels, 7)
    
    return torch.cat((centroid_y.unsqueeze(-1), centroid_x.unsqueeze(-1), rgb_mean, rgb_std, shape_moments, lbp_hist), dim=2), seq_masks