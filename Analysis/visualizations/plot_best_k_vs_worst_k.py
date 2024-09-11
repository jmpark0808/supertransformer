import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

tracer_dir = '/home/eddie/waterloo/TRACER/mask/DUTS/Test/images/'
ours_dir = '/home/eddie/waterloo/supertransformer/visualization/SP_SWINU/DUTS-TE/'

images_root = '/mnt/dragon/Datasets/DUTS/DUTS-TE/Image'
masks_root = '/mnt/dragon/Datasets/DUTS/DUTS-TE/Mask'


tracer_preds = {file.split('.')[0]:file.split('.')[1] for file in os.listdir(tracer_dir)}
ours_preds = {file.split('.')[0]:file.split('.')[1] for file in os.listdir(ours_dir)}
images = {file.split('.')[0]:file.split('.')[1] for file in os.listdir(images_root)[:]}
masks = {file.split('.')[0]:file.split('.')[1] for file in os.listdir(masks_root)}

results = {}
for image, ext in tqdm(images.items()):
    file_name = image+'.'+ext
    mask_name = image + '.' +masks[image]
    tracer_name = image + '.'+tracer_preds[image]
    ours_name = image + '.' + ours_preds[image]

    file_path = os.path.join(images_root, file_name)
    mask_path = os.path.join(masks_root, mask_name)
    tracer_path = os.path.join(tracer_dir, tracer_name)
    ours_path = os.path.join(ours_dir, ours_name)

    gt = np.array(Image.open(mask_path).convert('L').resize((320, 320)))/255.
    tracer = np.array(Image.open(tracer_path).convert('L'))/255.
    ours = np.array(Image.open(ours_path).convert('L'))/255.
    

    prec, recall = np.zeros((1, 256)), np.zeros((1, 256))
    pred = ours.reshape(1, -1)
    mask = gt.reshape(1, -1)
    thlist = np.linspace(0, 1 - 1e-10, 256)
    for j in range(256):
        y_temp = (pred >= thlist[j]).astype(np.float32)
        tp = (y_temp * mask).sum(axis=-1)
        # avoid prec becomes 0
        prec[:, j], recall[:, j] = (tp + 1e-10) / (y_temp.sum(axis=-1) + 1e-10), (tp + 1e-10) / (mask.sum(axis=-1) + 1e-10)


    beta_square = 0.3
    f_score = (1 + beta_square) * prec * recall / (beta_square * prec + recall)

    results[file_name] = np.max(f_score)

topk = 5
sorted_results = sorted(results, key=results.get)


worst_k = sorted_results[:topk]
best_k = sorted_results[-topk:]

fig1, ax1 = plt.subplots(topk, 4)
fig2, ax2 = plt.subplots(topk, 4)
worst_k_ind = 0 
best_k_ind = 0


for image, ext in tqdm(images.items()):
    
    file_name = image+'.'+ext
    if file_name not in worst_k and file_name not in best_k:
        continue

    mask_name = image + '.' +masks[image]
    tracer_name = image + '.'+tracer_preds[image]
    ours_name = image + '.' + ours_preds[image]

    file_path = os.path.join(images_root, file_name)
    mask_path = os.path.join(masks_root, mask_name)
    tracer_path = os.path.join(tracer_dir, tracer_name)
    ours_path = os.path.join(ours_dir, ours_name)
    
    gt = np.array(Image.open(mask_path).convert('L').resize((320, 320)))
    tracer = np.array(Image.open(tracer_path).convert('L').resize((320, 320)))
    ours = np.array(Image.open(ours_path).convert('L'))

    image = np.array(Image.open(file_path).convert('RGB').resize((320, 320)))
    

    if file_name in worst_k:
        ax1[worst_k_ind, 0].imshow(image)
        ax1[worst_k_ind, 1].imshow(gt, cmap='gray')
        ax1[worst_k_ind, 2].imshow(tracer, cmap='gray')
        ax1[worst_k_ind, 3].imshow(ours, cmap='gray')
        ax1[worst_k_ind, 0].axis('off')
        ax1[worst_k_ind, 1].axis('off')
        ax1[worst_k_ind, 2].axis('off')
        ax1[worst_k_ind, 3].axis('off')
        if worst_k_ind == 0:
            ax1[worst_k_ind, 0].set_title('Image')
            ax1[worst_k_ind, 1].set_title('Ground Truth')
            ax1[worst_k_ind, 2].set_title('TRACER')
            ax1[worst_k_ind, 3].set_title('Ours')
   
        
        ax1[worst_k_ind, 0].axis('off')
        ax1[worst_k_ind, 1].axis('off')
        ax1[worst_k_ind, 2].axis('off')
        ax1[worst_k_ind, 3].axis('off')

        worst_k_ind += 1

    else:
        ax2[best_k_ind, 0].imshow(image)
        ax2[best_k_ind, 1].imshow(gt, cmap='gray')
        ax2[best_k_ind, 2].imshow(tracer, cmap='gray')
        ax2[best_k_ind, 3].imshow(ours, cmap='gray')
        
        if best_k_ind == 0:
            ax2[best_k_ind, 0].set_title('Image')
            ax2[best_k_ind, 1].set_title('Ground Truth')
            ax2[best_k_ind, 2].set_title('TRACER')
            ax2[best_k_ind, 3].set_title('Ours')
        
        ax2[best_k_ind, 0].axis('off')
        ax2[best_k_ind, 1].axis('off')
        ax2[best_k_ind, 2].axis('off')
        ax2[best_k_ind, 3].axis('off')
        best_k_ind += 1


    
fig1.savefig('visualization/worstk.jpg')
fig2.savefig('visualization/bestk.jpg')












