import os
import matplotlib.pyplot as plt
import cv2 as cv
from skimage import io
from skimage.segmentation import mark_boundaries, slic
from skimage.measure import regionprops_table
import numpy as np
from PIL import Image
from tqdm import tqdm
import pickle

dataset_images = '/mnt/hdd/Datasets/DUTS/DUTS-TE/Image'
masks = '/mnt/hdd/Datasets/DUTS/DUTS-TE/Mask'

all_fscores = []
for file in tqdm(os.listdir(dataset_images)):
    name = file.split('.jpg')[0]
    image = os.path.join(dataset_images, name+'.jpg')
    mask = os.path.join(masks, name+'.png')

    img = Image.open(image)
    msk = Image.open(mask)
    img = img.convert('L').resize((300, 300))
    msk = msk.convert('L').resize((300, 300))
    img = np.array(img)
    msk = np.array(msk)
    
    ret, thresh = cv.threshold(img, 0, 255, cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
    thresh = thresh/255.
    # msk[msk>125] = 255
    # msk[msk<=125] = 0

    # empty_background = np.zeros_like(msk)

    # msk_boundaries = np.sum(mark_boundaries(empty_background, msk), axis=2)

    msk[msk<=125] = 0
    msk[msk>125] = 1
    


    msk = np.ravel(msk)
    thresh = np.ravel(thresh)
    y_temp = (thresh >= 0.5).astype(np.float)
    tp = np.sum((y_temp * msk))
    # avoid prec becomes 0
    prec, recall = (tp + 1e-10) / (np.sum(y_temp) + 1e-10), (tp + 1e-10) / (np.sum(msk) + 1e-10)
    beta_square = 0.3
    f_score = (1 + beta_square) * prec * recall / (beta_square * prec + recall)
    all_fscores.append(f_score)

print(np.mean(all_fscores))



