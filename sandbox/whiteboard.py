#from typing import Callable, Optional, Union
# from einops import rearrange
# import torch
# from torch import Tensor
# import torch.nn.functional as F
# import time
# from Blocks.performer2 import ViP, ViPEnc
# import numpy as np
# from fvcore.nn import FlopCountAnalysis, flop_count_table, parameter_count
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

#-------------------------------------------------------------------

# from dataset.superpixel import SPDataModule, SPDataset, DUTSDataset
# from torch.utils.data import DataLoader
# from tqdm import tqdm
# from Blocks.swinunet_mix_ape import SwinUTransformer
# from Blocks.performer2 import ViPU
# import os
# import time
# train_dir = '/mnt/dragon/Datasets/DUTS/DUTS-TR'
# test_dir = '/mnt/dragon/Datasets/DUTS/DUTS-TE'
# batch_size = 1
# num_workers = 20
# num_seg = 1024
# res = 224
# dataloader = 'SPFFT'
# compactness = 10
# coeff = 10
# ignore_phase = False

# image_list = np.array(sorted([os.path.join('{}/Image'.format(train_dir), f) for f in os.listdir('{}/Image'.format(train_dir))]))
# mask_list = np.array(sorted([os.path.join('{}/Mask'.format(train_dir), f) for f in os.listdir('{}/Mask'.format(train_dir))]))


# indices = np.array(list(range(len(image_list))))
# np.random.shuffle(indices)

# val_image_list = image_list[indices[int(len(image_list)*0.85):]]
# val_mask_list = mask_list[indices[int(len(mask_list)*0.85):]]

# tr_image_list = image_list[indices[:int(len(image_list)*0.85)]][:1000]
# tr_mask_list = mask_list[indices[:int(len(mask_list)*0.85)]][:1000]

# test_image_list = sorted([os.path.join('{}/Image'.format(test_dir), f) for f in os.listdir('{}/Image'.format(test_dir))])
# test_mask_list = sorted([os.path.join('{}/Mask'.format(test_dir), f) for f in os.listdir('{}/Mask'.format(test_dir))])


# dataset = SPDataset(tr_image_list, tr_mask_list, num_seg, res, compactness, True, dataloader, coeff, ignore_phase)
# # dataset = DUTSDataset(tr_image_list, tr_mask_list, num_seg, res, True)
        
# loader = DataLoader(
#                 dataset, batch_size=batch_size, 
#                 num_workers=num_workers, shuffle=True, pin_memory=True, drop_last=True)

# # model = SwinUTransformer(img_size=32, in_chans=28, patch_size=1, window_size=8,
# #                                        embed_dim=[32, 64, 128], depths=[2, 2, 6],
# #                                          num_heads=[2, 4, 8], mlp_ratio=4).cuda()
# model = ViPU(image_size=32, patch_size=1, dims=[32, 64, 128], depths=[2, 2, 6], heads=[2, 4, 8], mlp_ratio=4, channels=36).cuda()
# model.eval()

# all_times = []
# inp = torch.randn(1, 1024, 38).cuda()
# with torch.no_grad():
#     curr_time = time.time()
#     for _ in range(1000):

#     # for batch in tqdm(loader):
#         start = time.time()
#         model(inp)
        
#         curr_time = time.time()
#         all_times.append(curr_time-start)

# print(np.mean(all_times)*1000)
# assert(0)
        

# ---------------------------------------------------------------------------------------

# import torch
# import time
# from tqdm import tqdm

# num_seg = 1024
# img_size = 224
# seg = torch.arange(0, num_seg).unsqueeze(1).repeat(1, img_size*img_size//num_seg).reshape(1, 1, img_size, img_size).repeat(16, 1, 1, 1).long().cuda()
# img = torch.rand(16, 3, 224, 224).cuda()
# mask = torch.rand(16, 1, 224, 224).cuda()
# h, w = img.shape[-2:]
# xs = torch.arange(0, w, device=img.device).unsqueeze(0).float()
# ys = torch.arange(0, h, device=img.device).unsqueeze(1).float()
# xs = xs.repeat(h, 1)
# ys = ys.repeat(1, w)
# coord = torch.stack((xs, ys), 0).unsqueeze(0).repeat(img.size(0), 1, 1, 1)
# all_times = []
# for _ in tqdm(range(1)):
#     start = time.time()
#     label_onehot = F.one_hot(seg.reshape(seg.size(0), -1), num_seg).float()
#     area = label_onehot.sum(1).unsqueeze(-1)
#     area_input = area.detach().clone()

#     As = label_onehot.permute(0, 2, 1)
#     Bs = img.reshape(seg.size(0), 3, -1).permute(0, 2, 1)

    

#     Cs = coord.reshape(img.size(0), 2, -1).permute(0, 2, 1)
#     area[area==0] = torch.inf
#     colour = torch.einsum('bij,bjk->bik', As, Bs)/area
#     centroids = torch.einsum('bij,bjk->bik', As, Cs)/area
    

#     end = time.time()
#     all_times.append(end-start)

# print(np.array(all_times).mean())

# colour_method_1 = colour.detach().clone()
# centroids_method_1 = centroids.detach().clone()

# from torch_sparse import SparseTensor
# from torch_geometric.utils import scatter


# def batch_of_segs_to_sparse(seg, img, mask):
#     # seg (bs, 1, H, W)
#     # img (bs, 3, H, W)
#     # coord (bs, 2, H, W)
#     b, _, h, w = img.size()
#     xs = torch.arange(0, img_size, device=seg.device).unsqueeze(0).float()
#     ys = torch.arange(0, img_size, device=seg.device).unsqueeze(1).float()
#     xs = xs.repeat(img_size, 1)
#     ys = ys.repeat(1, img_size)
#     coord = torch.stack((xs, ys), 0).unsqueeze(0)
#     coord = coord.repeat(b, 1, 1, 1)

#     shift = torch.arange(0, b, device=seg.device).repeat_interleave(h*w)*num_seg
    
#     seg = seg.reshape(-1)+shift
#     img = img.permute(0, 2, 3, 1).reshape(-1, 3)
#     mask = mask.reshape(-1)+shift
#     area = torch.ones_like(seg)
    
#     coord = coord.permute(0, 2, 3, 1).reshape(-1, 2)
#     # seg = SparseTensor(row=seg, col=torch.arange(0, seg.size(0), device='cuda'))
#     # edge_index = torch.stack((torch.arange(0, seg.size(0), device='cuda'), seg), dim=0)
    
    

#     colour = scatter(img, seg, reduce='sum',  dim_size=num_seg*b)
#     centroid = scatter(coord, seg, reduce='mean', dim_size=num_seg*b)
#     area = scatter(area, seg, reduce='sum', dim_size=num_seg*b)
#     seq_mask = scatter(mask, seg, reduce='mean', dim_size=num_seg*b)
#     # colour = seg.matmul(img) # BS*H*W x 3
#     # centroid = seg.matmul(coord) # BS*H*W x 2
    
    
#     colour = colour.reshape(b, num_seg, 3)
#     centroid = centroid.reshape(b, num_seg, 2)
#     area = area.reshape(b, num_seg)
#     seq_mask = seq_mask.reshape(b, num_seg)
#     return colour, centroid, area, seq_mask





# all_times = []
# for _ in tqdm(range(1)):
#     start = time.time()

#     colour, centroids, _, _= batch_of_segs_to_sparse(seg, img, mask)


#     end = time.time()
#     all_times.append(end-start)


# print(np.array(all_times).mean())

# print(torch.norm((colour-colour_method_1).reshape(-1), p=2), torch.norm((centroids-centroids_method_1).reshape(-1), p=2))

# import matplotlib.pyplot as plt

# plt.plot(centroids[0, :, 0].detach().cpu().numpy())
# plt.show()



# from fast_slic.avx2 import SlicAvx2
# import numpy as np
# from skimage.measure import regionprops_table
# import matplotlib.pyplot as plt
# from skimage.segmentation import mark_boundaries
# img_np = (np.ones([320, 320, 3])*255).astype(np.uint8)
# slic = SlicAvx2(num_components=1024, compactness=10, min_size_factor=0.)
# segments = slic.iterate(img_np)

# regions = regionprops_table(segments, intensity_image=img_np, properties=('label', 'centroid'))#, polarize])
# centers_y = regions['centroid-0']
# centers_x = regions['centroid-1']
# plt.imshow(mark_boundaries(img_np, segments))
# plt.scatter(centers_x, centers_y, c='blue', s=30)
# for ind, (x, y) in enumerate(zip(centers_x, centers_y)):
#     plt.text(x, y, str(regions['label'][ind]))
# plt.show()

#--------------------------------------------------------------------------------------

# import numpy as np
# import matplotlib.pyplot as plt
# tr_images = np.load('/mnt/dragon/Datasets/PASCAL/archive (1)/images.npy')
# tr_masks = np.load('/mnt/dragon/Datasets/PASCAL/archive (1)/masks.npy')
# # te_images = np.load('/mnt/dragon/Datasets/PASCAL/archive (1)/pascals_test_images.npy')
# # te_masks = np.load('/mnt/dragon/Datasets/PASCAL/archive (1)/pascals_test_masks.npy')

# export_image_dir = '/mnt/dragon/Datasets/PASCAL/Image'
# export_mask_dir = '/mnt/dragon/Datasets/PASCAL/Mask'


# tr_images = tr_images.reshape(-1, 256, 256, 3)
# tr_masks = tr_masks.reshape(-1, 256, 256)
# # te_images = te_images.reshape(-1, 256, 256, 3)
# # te_masks = te_masks.reshape(-1, 256, 256)

# # all_images = np.concatenate((tr_images, te_images), axis=0)
# # all_masks = np.concatenate((tr_masks, te_masks), axis=0)
# all_images = tr_images
# all_masks = tr_masks

# import cv2
# import os
# for idx, image in enumerate(all_images):
#     cv2.imwrite(os.path.join(export_image_dir, f'{idx}.png'), image[:, :, ::-1])

# for idx, mask in enumerate(all_masks):
#     cv2.imwrite(os.path.join(export_mask_dir, f'{idx}.png'), mask)
    
#-------------------------------------------------------------------------------------------



# import os
# from PIL import Image
# import numpy as np
# from tqdm import tqdm
# from torchvision.datasets import ImageFolder
# from torch.utils.data import DataLoader
# from torchvision import transforms
# imagenet_dir = '/mnt/dragon/Datasets/DUTS/DUTS-TR/Image'
# # dataset = ImageFolder(imagenet_dir, transform=transforms.Compose([transforms.ToTensor()]))
# # loader = DataLoader(dataset, batch_size=1, num_workers=20)
# all_heights = []
# all_widths = []

# for img in os.listdir(imagenet_dir):
#     img_path = os.path.join(imagenet_dir, img)
#     img = np.array(Image.open(img_path).convert('RGB'))
#     height, width, c = img.shape
#     all_heights.append(height)
#     all_widths.append(width)

# # for images, labels in tqdm(loader):
# #     _, _, height, width = images.size()
# #     all_heights.append(height)
# #     all_widths.append(width)
    

# print(np.mean(all_heights))
# print(np.mean(all_widths))



#-----------------------------------------------------------------------------

# import os
# import numpy as np
# import matplotlib.pyplot as plt
# main_dir = '/mnt/dragon/Datasets/sp_train'
# files = os.listdir(main_dir)
# count = 0
# coeffs = np.zeros([55])
# for file in files:
#     if 'target' not in file:
#         features = np.load(os.path.join(main_dir, file))
        
#         amplitude = features[:, 9:9+55]
#         coeffs += amplitude.sum(0)
#         count += amplitude.shape[0]

# print(coeffs/count)
# plt.bar(list(range(55)), coeffs/count)
# plt.show()

        

# -----------------------------------------------------------

# import torch
# for _ in range(1000):
#     pred = torch.rand(30, 1024).cuda()
#     mask = torch.rand(30, 1024).cuda()

#     prec, recall = torch.zeros(pred.size(0), 10).cuda(), torch.zeros(pred.size(0), 10).cuda()

#     thlist = torch.linspace(0, 1 - 1e-10, 10).cuda()
#     for j in range(10):
#         y_temp = (pred >= thlist[j]).float()
#         tp = (y_temp * mask).sum(dim=-1)
#         # avoid prec becomes 0
#         prec[:, j], recall[:, j] = (tp + 1e-10) / (y_temp.sum(dim=-1) + 1e-10), (tp + 1e-10) / (mask.sum(dim=-1) + 1e-10)
                
#     prec_mean = prec.mean(0)
#     prec_sum = prec.sum(0)/30
#     recall_mean = recall.mean(0)
#     recall_sum = recall.sum(0)/30      


#     beta_square = 0.3
#     f_score_mean = (1 + beta_square) * prec_mean * recall_mean / (beta_square * prec_mean + recall_mean)
#     f_score_sum = (1 + beta_square) * prec_sum * recall_sum / (beta_square * prec_sum + recall_sum)

#     print(torch.sum(torch.abs(f_score_mean-f_score_sum)))


# -------------------------------------------------
# import torch

# a = torch.zeros(100, 2)
# ind = 0 
# for i in range(0, 10):
#     for j in range(0, 10):
#         a[ind] = torch.tensor([i, j])
#         ind +=1
# print(a)
# b = torch.nn.LayerNorm(2)
# c = b(a)
# print(c)

        
