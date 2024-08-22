from typing import Callable, Optional, Union
from einops import rearrange
import torch
from torch import Tensor
import torch.nn.functional as F
import time
from Blocks.performer2 import ViP, ViPEnc
import numpy as np
from fvcore.nn import FlopCountAnalysis, flop_count_table, parameter_count
# model = ViP(image_size=32, patch_size=1, dim=32, depth=6, heads=2, mlp_dim=32*4, channels=16, dim_head=16).cuda()
# model.eval()




# dummy_input = torch.randn(1, 1024, 38).cuda()
# l = []
# with torch.no_grad():
#     for i in range(1000):
#         start = time.time()
#         model(dummy_input)
#         end = time.time()
#         l.append(end-start)

# print(np.mean(l))


# model = ViPEnc(image_size=32, patch_size=1, dims=[32, 64, 128, 256], depths=[2, 2, 6, 2], heads=[2,4, 8, 16], mlp_ratio=4, channels=16).cuda()
# model.eval()

# l = []
# with torch.no_grad():
#     for i in range(1000):
#         start = time.time()
#         model(dummy_input)
#         end = time.time()
#         l.append(end-start)

# print(np.mean(l))


# linear = torch.nn.Linear(256, 256)
# class conv(torch.nn.Module):
#     def __init__(self) -> None:
#         super().__init__()
#         self.conv = torch.nn.Conv1d(256, 256*8, 1, groups=8)

#     def forward(self, x):
#         x = x.permute(0, 2, 1)
#         x = self.conv(x)
#         x = rearrange(x, 'b (c g) n->b c g n', g=8)
#         x = x.mean(2)
#         return x
    
# conv_model = conv()

# params_linear = parameter_count(linear)['']
# params_conv = parameter_count(conv_model)['']

# inp = torch.randn([1, 1024, 256])
# flops_linear = FlopCountAnalysis(linear, inp)
# flops_conv = FlopCountAnalysis(conv_model, inp)
# flops_linear = flops_linear.total()
# flops_conv = flops_conv.total()

# print(params_linear, flops_linear)
# print(params_conv, flops_conv)

# from Blocks.DiffSLIC import DiffSLIC
# from PIL import Image
# # slic_fn = DiffSLIC(n_spixels=1024, n_iter=10, tau=1, candidate_radius=1, stable=True, normalize=False, compactness=1)
# from torchvision.transforms import ToTensor
# import matplotlib.pyplot as plt
# from skimage.segmentation import mark_boundaries
# rgb_img = Image.open('/mnt/dragon/Datasets/DUTS/DUTS-TE/Image/ILSVRC2012_test_00000003.jpg').resize((224, 224))
# tt = ToTensor()
# rgb_img = tt(rgb_img).unsqueeze(0)

# # features, spix2pix_assign, pix2spix_assign, hard_assignment = slic_fn(rgb_img)
# # hard_assignment = hard_assignment.long().detach().numpy()-1
# # print(features[:, :, 0, 0])

# from torch_kmeans import SoftKMeans

# model = SoftKMeans()

# # x = torch.randn(1, 1024, 32)
# result = model(rgb_img.reshape(1, 3, -1).permute(0, 2, 1), k=256)
# print(result.soft_assignment.size())
# plt.imshow(mark_boundaries(rgb_img[0].permute(1, 2, 0).detach().numpy(), result.labels.reshape(1, 224, 224).detach().numpy().squeeze()))
# # plt.imshow(features.reshape(5,1024).permute(1, 0).detach().numpy()[hard_assignment.reshape(-1), :].reshape(320, 320, 5)[:, :, :3])
# plt.show()


from dataset.superpixel import SPDataModule, SPDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
train_dir = '/mnt/dragon/Datasets/DUTS/DUTS-TR'
test_dir = '/mnt/dragon/Datasets/DUTS/DUTS-TE'
batch_size = 16
num_workers = 20
num_seg = 1024
res = 224
dataloader = 'SP'
compactness = 10
coeff = 10
ignore_phase = False

image_list = np.array(sorted([os.path.join('{}/Image'.format(train_dir), f) for f in os.listdir('{}/Image'.format(train_dir))]))
mask_list = np.array(sorted([os.path.join('{}/Mask'.format(train_dir), f) for f in os.listdir('{}/Mask'.format(train_dir))]))


indices = np.array(list(range(len(image_list))))
np.random.shuffle(indices)

val_image_list = image_list[indices[int(len(image_list)*0.85):]]
val_mask_list = mask_list[indices[int(len(mask_list)*0.85):]]

tr_image_list = image_list[indices[:int(len(image_list)*0.85)]]
tr_mask_list = mask_list[indices[:int(len(mask_list)*0.85)]]

test_image_list = sorted([os.path.join('{}/Image'.format(test_dir), f) for f in os.listdir('{}/Image'.format(test_dir))])
test_mask_list = sorted([os.path.join('{}/Mask'.format(test_dir), f) for f in os.listdir('{}/Mask'.format(test_dir))])


dataset = SPDataset(tr_image_list, tr_mask_list, num_seg, res, compactness, True, dataloader, coeff, ignore_phase)
        
loader = DataLoader(
                dataset, batch_size=batch_size, 
                num_workers=num_workers, shuffle=True, pin_memory=True, drop_last=True)

for batch in tqdm(loader):
    pass