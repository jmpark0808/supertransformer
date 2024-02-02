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
from collections import defaultdict

dataset_images = '/mnt/hdd/Datasets/DUTS/DUTS-TE/Image'
masks = '/mnt/hdd/Datasets/DUTS/DUTS-TE/Mask'


def flow_temp(s, v, U, L, image):
    u_temp = np.max(np.concatenate((U[s], image[v]), axis=0), axis=0) #(3, )
    l_temp = np.min(np.concatenate((L[s], image[v]), axis=0), axis=0) #(3, )
    d_temp = np.mean(u_temp-l_temp)
    return d_temp

def flow(s, v, D, U, L, image):
    U[v] = np.max(np.concatenate((U[s], image[v]), axis=0), axis=0) #(3, )
    L[v] = np.min(np.concatenate((L[s], image[v]), axis=0), axis=0) #(3, )
    D[v] = np.mean(U[v]-L[v])
    return D, U, L

def nn(image, i, j, M, U, L):
    up = (i-1, j)
    down = (i+1, j)
    left = (i, j-1)
    right = (i, j+1)
    up_d = 300
    down_d = 300
    left_d = 300
    right_d = 300
    d = {}
    if M[up] == 1:
        up_d = flow_temp(up, (i, j), U, L, image)
        d[up] = up_d
        
    if M[down] == 1:
        down_d = flow_temp(down, (i, j), U, L, image)
        d[down] = down_d

    if M[left] == 1:
        left_d = flow_temp(left, (i, j), U, L, image)
        d[left] = left_d

    if M[right] == 1:
        right_d = flow_temp(right, (i, j), U, L, image)
        d[right] = right_d

    d = {k: v for k, v in sorted(d.items(), key=lambda item: item[1])}
    return list(d.keys())[0]

def waterflow(image):
    # 0 = droughty
    # 1 = flooded
    # 2 = waiting
    Q = defaultdict(list)
    M = np.ones((image.shape[0], image.shape[1]))
    M[1:-1, 1:-1] = 0
    U = np.copy(image)
    L = np.copy(image)
    D = np.zeros((image.shape[0], image.shape[1]))
    for i in range(1, image.shape[1]-1):
        for j in range(1, image.shape[1]-1):
            if i == 1 or j == 1 or i == image.shape[1]-2 or j == image.shape[1]-2:
                s = nn(image, i, j, M, U, L)
                D, U, L = flow(s, (i, j), D, U, L, image)
                Q[D[i, j]].append((i,j))
                M[i, j] = 2
            else:
                pass

    r = Q[sorted(Q)[0]][0]
    min_dist = sorted(Q)[0]
    while len(Q) != 0:
        if M[r] == 1:
            Q[min_dist].pop(0)
            if len(Q[min_dist]) == 0:
                Q.pop(min_dist)
                try:
                    min_dist = sorted(Q)[0]
                    r = Q[min_dist][0]
                except:
                    pass
            else:
                r = Q[min_dist][0]

            continue
        else:
            M[r] = 1

        neighbours = [(r[0]+1,r[1]), (r[0]-1, r[1]), (r[0], r[1]+1), (r[0], r[1]-1)]
        for neighbour in neighbours:
            if M[neighbour] == 0:
                D, U, L = flow(r, neighbour, D, U, L, image)
                Q[D[neighbour]].append(neighbour)
                min_dist = sorted(Q)[0]
                r = Q[min_dist][0]
                M[neighbour] = 2
            elif M[neighbour] == 2 and D[neighbour] > D[r]:
                if D[neighbour] > flow_temp(r, neighbour, U, L, image):
                    D, U, L = flow(r, neighbour, D, U, L, image)
                    Q[D[neighbour]].append(neighbour)
                    min_dist = sorted(Q)[0]
                    r = Q[min_dist][0]


    return D








all_fscores = []
precs = []
recalls = []
for file in tqdm(os.listdir(dataset_images)):
    name = file.split('.jpg')[0]
    image = os.path.join(dataset_images, name+'.jpg')
    mask = os.path.join(masks, name+'.png')

    img = Image.open(image)
    msk = Image.open(mask)
    img = img.convert('RGB').resize((300, 300))
    msk = msk.convert('L').resize((300, 300))
    img = np.array(img)
    msk = np.array(msk)
    
    D = waterflow(img)
    D = D/np.max(D)


    msk[msk<=125] = 0
    msk[msk>125] = 1
    


    msk = np.ravel(msk)
    thresh = np.ravel(D)

    prec, recall = np.zeros(256), np.zeros(256)
    thlist = np.linspace(0, 1 - 1e-10, 256)
    for j in range(256):
        y_temp = (thresh >= thlist[j]).astype(float)
        tp = np.sum((y_temp * msk), axis=-1)
        # avoid prec becomes 0
        prec[j], recall[ j] = (tp + 1e-10) / (np.sum(y_temp, axis=-1) + 1e-10), (tp + 1e-10) / (np.sum(msk, axis=-1) + 1e-10)
    # (batch, threshold)
    precs.append(prec)
    recalls.append(recall)



prec = np.mean(np.stack(precs, axis=0), axis=0)
recall = np.mean(np.stack(recalls, axis=0), axis=0)
beta_square = 0.3
f_score = (1 + beta_square) * prec * recall / (beta_square * prec + recall)
print(np.max(f_score))




