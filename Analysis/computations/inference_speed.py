# load image 
import sys
from threading import local
sys.path.insert(0, '/home/eddie/waterloo/supertransformer')
from torchvision import transforms, datasets
import torch
from PIL import Image
from Blocks import blocks
import numpy as np
from fast_slic.avx2 import SlicAvx2
from skimage.measure import regionprops_table
from skimage.segmentation import mark_boundaries
from skimage.feature import local_binary_pattern
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os
from numpy_superpixel import SLICProcessor
import time
from sklearn.metrics.pairwise import euclidean_distances
from tqdm import tqdm
from scipy import sparse as sp
from dataset.constants import *
from scipy.spatial.distance import pdist, squareform
import cv2
from Models.SP_TFM import SP_TFM_REL
from Models.SP_GAT import SP_GAT
from Models.SP_CNN import SP_CNN_LIN
from Models.SP_TFM import SP_TFM_FFT
from dataset.attributes import *
from dataset.superpixel import SPDataset
from torch.utils.data import DataLoader


def fourier_descriptors(region):
    region = (region*255).astype(np.uint8)
    contour, hierarchy = cv2.findContours(region, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    points = contour[0][:, 0, :]
    xi, yi = resample_2d(points, RESAMPLE_POINTS)
    contour_array = np.stack((xi, yi), axis=1)


    contour_complex = np.empty(contour_array.shape[:-1], dtype=complex)
    contour_complex.real = contour_array[:, 0]
    contour_complex.imag = contour_array[:, 1]
    fourier_result = np.fft.fft(contour_complex)

    fourier_result_front = fourier_result[1:1+10//2]
    fourier_result_back = fourier_result[-10//2-1:-1]
    fourier_result = np.concatenate((fourier_result_front, fourier_result_back), axis=0)

    amp = abs(fourier_result)
    phase = np.arctan2(fourier_result.imag, fourier_result.real)

    # return np.array(amp)
    return np.concatenate((amp, phase))


data_dir = '/mnt/hdd/Datasets/DUTS/DUTS-TR/Image/'
all_distances = []
heights = []
widths = []
dataset = SPDataset('/mnt/hdd/Datasets/DUTS/DUTS-TE', 625, 224, 10, False, 'SPFFT', 10, False)
dataloader = DataLoader(
                dataset, batch_size=1, 
                num_workers=14, shuffle=False, pin_memory=True)

tp = time.time()
for batch in dataloader:
    print((time.time()-tp)*1000.)
    tp = time.time()
    

model = SP_TFM_FFT(146, 16, 8, 6, 0).cuda()

counter = 0
collect = []
for file in tqdm(os.listdir(data_dir)):
    model.eval()
    img = Image.open(os.path.join(data_dir, file))
    img = img.convert('RGB')
    img = img.resize((300, 300), resample=Image.BILINEAR)

    img_np = (np.array(img)/255).astype(np.uint8)

    num_seg = 625


    img_size = img_np.shape[1]
    slic = SlicAvx2(num_components=num_seg, compactness=10.0, min_size_factor=0)
    start = time.time()
    segments = slic.iterate(img_np)+1
    regions = regionprops_table(segments, intensity_image=img_np, properties=(['label']), extra_properties=[fourier_descriptors])#, polarize]
        
    end = time.time()
    print('SLIC', (end-start)*1000)
    

    
    # dummy_input = torch.ones([1, 625, 146]).cuda()
    # # adj = torch.ones([1, 100, 500]).cuda()
    # device = 0
    # with torch.no_grad():
    #     start = time.time()
    #     model(dummy_input)
    #     end = time.time()
    #     # print('Model', (end-start)*1000)
    #     collect.append((end-start)*1000)

    # model.train()
    # model.cpu()
    # a = torch.cuda.memory_allocated(device)
    # model.to(device)
    # b = torch.cuda.memory_allocated(device)
    # model_memory = b - a
    # output = model(dummy_input)
    # c = torch.cuda.memory_allocated(device)
    # amp_multiplier = 1
    # forward_pass_memory = (c - b)*amp_multiplier
    # total_memory = model_memory+forward_pass_memory
    # end = time.time()
    # print(total_memory)
    

    # model = SP_GAT(11, 16, 0, 8, 16, 0.2).cuda()
    # dummy_input = torch.ones([1, 625, 11]).cuda()
    # adj = torch.ones([1, 625, 625]).cuda()

    # start = time.time()
    # model([dummy_input, adj])
    # end = time.time()


    # model = SP_CNN_LIN().cuda()
    # dummy_input = torch.ones([1, 625, 9]).cuda()

    # start = time.time()
    # model(dummy_input)
    # end = time.time()
    
    # start = time.time()
    # eye = np.eye(625)
    # A = np.ones([625, 625])
    # N = sp.diags(np.sum(A, axis=0).clip(1) ** -0.5, dtype=float)
    # L = eye - N * A * N


    # # Eigenvectors with numpy
    # EigVal, EigVec = np.linalg.eig(L)
    # idx = EigVal.argsort() # increasing order
    # EigVal, EigVec = EigVal[idx], np.real(EigVec[:,idx])
    # pos_enc = torch.from_numpy(EigVec[:,1:]).float() 
    # end = time.time()
    # print((end-start)*1000)

    counter += 1
    if counter == 1000:
        break

print(np.mean(collect))

# print(np.min(heights), np.max(heights))
# print(np.min(widths), np.max(widths))