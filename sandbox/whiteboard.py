
import torch
import matplotlib.pyplot as plt
import numpy as np

def generate_grid_segmentation(h=224, w=224, patch_size=4):
    """
    Generate a dummy segmentation map where each superpixel is a square patch.
    The labels range from 0 to (H // patch_size) * (W // patch_size) - 1.

    Returns:
        seg_map: (H, W) LongTensor with labels from 0 to K-1
    """
    assert h % patch_size == 0 and w % patch_size == 0, "Patch size must divide dimensions evenly"

    n_rows = h // patch_size
    n_cols = w // patch_size
    K = n_rows * n_cols

    # Create label grid (n_rows, n_cols)
    labels = torch.arange(K).view(n_rows, n_cols)  # (56, 56)

    # Repeat each label in both H and W directions
    seg_map = labels.repeat_interleave(patch_size, dim=0).repeat_interleave(patch_size, dim=1)  # (224, 224)

    return seg_map.long()

def visualize_fft(mag_crop, phase_crop, batch_idx=0, label_idx=0):
    """
    Visualize FFT magnitude and phase for a specific batch and label index.
    
    Args:
        mag_crop: (B, K, k, k) magnitude tensor
        phase_crop: (B, K, k, k) phase tensor
        batch_idx: which image in the batch to visualize
        label_idx: which label in the segmentation to visualize
    """
    # Convert to numpy
    mag = mag_crop[batch_idx, label_idx].cpu().numpy()
    phase = phase_crop[batch_idx, label_idx].cpu().numpy()

    # Log scale for better dynamic range (add small epsilon to avoid log(0))
    mag_log = np.log1p(mag)

    # Create figure
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))

    axs[0].imshow(mag_log, cmap='gray')
    axs[0].set_title(f'Magnitude Spectrum (log1p), B={batch_idx}, L={label_idx}')
    axs[0].axis('off')

    im = axs[1].imshow(phase, cmap='twilight', vmin=-np.pi, vmax=np.pi)
    axs[1].set_title(f'Phase Spectrum, B={batch_idx}, L={label_idx}')
    axs[1].axis('off')

    plt.colorbar(im, ax=axs[1], orientation='vertical', fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()

# segmentation = torch.randint(0, 3136, (4, 224, 224)).cuda()  # B=4, H=W=64, K=5
segmentation = generate_grid_segmentation().unsqueeze(0).repeat(1, 1, 1)
B, H, W = segmentation.shape
device = segmentation.device
N = H * W
num_labels = 3136
k = 10


# Flatten spatial dimension
seg_flat = segmentation.view(B, -1)  # (B, N)

# Create one-hot masks via scatter
one_hot = torch.zeros(B, num_labels, N, device=device, dtype=torch.float32)
one_hot.scatter_(1, seg_flat.unsqueeze(1), 1.0)  # (B, K, N)

# Reshape to binary masks: (B, K, H, W)
masks = one_hot.view(B, num_labels, H, W)

# Apply 2D FFT to each mask (parallel over B and K)
fft_complex = torch.fft.fft2(masks)  # (B, K, H, W)
fft_shifted = torch.fft.fftshift(fft_complex, dim=(-2, -1))  # shift DC to center


fft_mag = torch.abs(fft_shifted)  # (B, K, H, W)
fft_phase = torch.angle(fft_shifted)  # (B, K, H, W)

visualize_fft(fft_mag, fft_phase, 0, 0)


# Center crop
h_center = H // 2
w_center = W // 2
half_k = k // 2

crop_h = slice(h_center - half_k, h_center + half_k)
crop_w = slice(w_center - half_k, w_center + half_k)

fft_mag_crop = fft_mag[:, :, crop_h, crop_w]  # (B, K, k, k)
fft_phase_crop = fft_phase[:, :, crop_h, crop_w]  # (B, K, k, k)