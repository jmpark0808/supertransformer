from Blocks.swin_encoder_rope import SwinTransformer
from Blocks.swinunet_mix_rope import SwinUTransformer
import torch


res = 56
coeff = 10 
input_dim = 44
window_size = 7
dims = [32, 64, 128, 512]
depths = [2, 2, 6, 2]
heads = [2, 4, 8, 32]
mlp_ratio = 2
classes = 1000
dropout_edge = 0 
dp = 0.1
dropout = 0
load = '/home/eddie/sp_imgnet_ogswin_ape_rope_56370392.ckpt'

supert_imgnet = SwinTransformer(img_size=res, coeff=coeff, in_chans=input_dim, patch_size=1, window_size=window_size,
                                       embed_dim=dims, depths=depths,
                                         num_heads=heads, mlp_ratio=mlp_ratio, num_classes=classes, attn_drop_rate=dropout_edge, 
                                         qkv_bias=False, drop_path_rate=dp, drop_rate=dropout)
dims = [32, 64, 128]
depths = [2, 2, 6]
heads = [2, 4, 8]

supert_swinum = SwinUTransformer(img_size=res, in_chans=input_dim, patch_size=1, window_size=window_size,
                                       embed_dim=dims, depths=depths,
                                         num_heads=heads, mlp_ratio=mlp_ratio, attn_drop_rate=dropout_edge, drop_rate=dropout,
                                         drop_path_rate=dp) 

ckpt = torch.load(load)
for key in list(ckpt['state_dict'].keys()):
    ckpt['state_dict'][key.replace('supert.', '')] = ckpt['state_dict'].pop(key)
supert_imgnet.load_state_dict(ckpt['state_dict'])



checkpoint = torch.load(load)
for key in list(checkpoint['state_dict'].keys()):
    checkpoint['state_dict'][key.replace('supert.', '')] = checkpoint['state_dict'].pop(key)

supert_swinum.load_state_dict(checkpoint['state_dict'], strict=False)


random_input = torch.randn(1, 46, 56, 56)

imgnet_output = supert_imgnet(random_input)
swinum_output = supert_swinum(random_input)

print(torch.sum(torch.abs(imgnet_output-swinum_output)))



sod_file = '/home/eddie/Datasets/DUTS/DUTS-TR/SPFFFT/ILSVRC2012_val_00000744_features.npy'
imgnet_file = '/home/eddie/Datasets/sp_test/ILSVRC2012_val_00000744.npy'

import numpy as np

sod_np = np.load(sod_file)
imgnet_np = np.load(imgnet_file)


print(np.sum(np.abs(sod_np-imgnet_np)))
print(sod_np.shape)
print(imgnet_np.shape)


sod_mins = np.min(sod_np, axis=0)
sod_maxs = np.max(sod_np, axis=0)

imgnet_mins = np.min(imgnet_np, axis=0)
imgnet_maxs = np.max(imgnet_np, axis=0)

import matplotlib.pyplot as plt



# fig, ax = plt.subplots(1, 2)
# ax[0].plot(sod_mins)
# ax[0].plot(sod_maxs)
# ax[1].plot(imgnet_mins)
# ax[1].plot(imgnet_maxs)
# plt.show()


duts_img = '/home/eddie/Datasets/DUTS/DUTS-TR/Image/ILSVRC2012_val_00000744.jpg'
imgnet_img = '/home/eddie/Datasets/ImageNet/val/n01847000/ILSVRC2012_val_00000744.JPEG'

from PIL import Image

duts_img = Image.open(duts_img)
imgnet_img = Image.open(imgnet_img)

duts_img_np = np.array(duts_img)
imgnet_img_np = np.array(imgnet_img)

print(duts_img_np.shape)
print(imgnet_img_np.shape)





from PIL import Image
import numpy as np
import os

def mse(image1, image2):
    """Calculate Mean Squared Error between two images"""
    arr1 = np.array(image1).astype("float")
    arr2 = np.array(image2).astype("float")
    return np.mean((arr1 - arr2) ** 2)

def load_and_resize(path, size):
    """Load an image and resize to a common size"""
    return Image.open(path).convert("RGB").resize(size)

def find_closest_image(ref_path, image_paths):
    # Load reference image and get size
    ref_img = Image.open(ref_path).convert("RGB")
    ref_size = ref_img.size

    ref_img = ref_img.resize(ref_size)
    min_error = float("inf")
    closest_image_path = None

    for path in image_paths:
        try:
            img = Image.open(path).convert("RGB").resize(ref_size)
            error = mse(ref_img, img)
            print(f"MSE between {os.path.basename(ref_path)} and {os.path.basename(path)}: {error:.2f}")
            if error < min_error:
                min_error = error
                closest_image_path = path
        except Exception as e:
            print(f"Error processing {path}: {e}")

    return closest_image_path

ref_image_path = '/home/eddie/Pictures/Screenshots/Screenshot from 2025-06-19 10-42-29.png'
image_folder = '/home/eddie/Datasets/DUTS/DUTS-TR/Image'
image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

closest = find_closest_image(ref_image_path, image_paths)
if closest:
    print(f"\nClosest image to '{ref_image_path}' is: {closest}")