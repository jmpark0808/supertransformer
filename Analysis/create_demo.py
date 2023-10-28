import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
from skimage.segmentation import slic
from skimage.measure import regionprops_table
from skimage.segmentation import mark_boundaries

monkey_dir = '/mnt/hdd/Datasets/SegTrackv2/JPEGImages/monkey'
girl_dir = '/mnt/hdd/Datasets/SegTrackv2/JPEGImages/girl'
soldier_dir = '/mnt/hdd/Datasets/SegTrackv2/JPEGImages/soldier'
bird_dir = '/mnt/hdd/Datasets/SegTrackv2/JPEGImages/bird_of_paradise'

monkey_gt_dir = '/mnt/hdd/Datasets/SegTrackv2/GroundTruth/monkey'
girl_gt_dir = '/mnt/hdd/Datasets/SegTrackv2/GroundTruth/girl'
soldier_gt_dir = '/mnt/hdd/Datasets/SegTrackv2/GroundTruth/soldier'
bird_gt_dir = '/mnt/hdd/Datasets/SegTrackv2/GroundTruth/bird_of_paradise'

monkey_tracer_dir = '/home/eddie/waterloo/TRACER/mask/monkey'
girl_tracer_dir = '/home/eddie/waterloo/TRACER/mask/girl'
soldier_tracer_dir = '/home/eddie/waterloo/TRACER/mask/soldier'
bird_tracer_dir = '/home/eddie/waterloo/TRACER/mask/bird_of_paradise'

monkey_sf_dir = '/home/eddie/waterloo/supertransformer/visualization/SP_TFM/Monkey'
girl_sf_dir = '/home/eddie/waterloo/supertransformer/visualization/SP_TFM/Girl'
soldier_sf_dir = '/home/eddie/waterloo/supertransformer/visualization/SP_TFM/Solider'
bird_sf_dir = '/home/eddie/waterloo/supertransformer/visualization/SP_TFM/Bird_of_paradise'

all_objects = [monkey_dir, girl_dir, soldier_dir, bird_dir]
ground_truths = [monkey_gt_dir, girl_gt_dir, soldier_gt_dir, bird_gt_dir]
tracers = [monkey_tracer_dir, girl_tracer_dir, soldier_tracer_dir, bird_tracer_dir]
sfs = [monkey_sf_dir, girl_sf_dir, soldier_sf_dir, bird_sf_dir]

def cum_mean(arr):
    cum_sum = np.cumsum(arr, axis=0)    
    for i in range(cum_sum.shape[0]):       
        if i == 0:
            continue        
        cum_sum[i] =  cum_sum[i] / (i + 1)
    return cum_sum

for object, gt, tracer, sf in zip(all_objects, ground_truths, tracers, sfs):
    objects = sorted([x for x in os.listdir(object) if not 'smooth' in x])
    objects_sp =  sorted([x for x in os.listdir(object) if 'smooth' in x])
    gts = sorted(os.listdir(gt))
    tr = sorted(os.listdir(tracer))
    s = sorted(os.listdir(sf))

    precs_d = []
    precs_e = []
    precs_max = []
    recalls_d = []
    recalls_e = []
    recalls_max = []

    f_scores_d = []
    f_scores_e = []
    f_scores_max = []

    for a, b, c, d, e in zip(objects, objects_sp, gts, tr, s):
        a_img = cv2.imread(os.path.join(object, a))
        
        b_img = cv2.imread(os.path.join(object, b))

       

        c_img = cv2.imread(os.path.join(gt, c), cv2.IMREAD_GRAYSCALE)
        d_img = cv2.imread(os.path.join(tracer, d), cv2.IMREAD_GRAYSCALE)
        e_img = cv2.imread(os.path.join(sf, e), cv2.IMREAD_GRAYSCALE)

        a_img = cv2.resize(a_img, (224, 224))
        b_img = cv2.resize(b_img, (224, 224))
        c_img = cv2.resize(c_img, (224, 224))
        d_img = cv2.resize(d_img, (224, 224))
        e_img = cv2.resize(e_img, (224, 224))
        
        img_np = np.array(a_img[...,::-1])
        segments_256 = slic(img_np, n_segments=256,
            compactness=10,
            max_num_iter=3,
            convert2lab=True,
            enforce_connectivity=True,
            slic_zero=False)
        
        regions = regionprops_table(segments_256, intensity_image=img_np, properties=('label','intensity_mean',
                                                                                'coords'))#, polarize])

        smoothed_images = np.zeros_like(img_np)
        for coord, r, g, b in zip(regions['coords'], regions['intensity_mean-0'], regions['intensity_mean-1'], regions['intensity_mean-2']):
            for co in coord:
                smoothed_images[co[0], co[1], 0] = r
                smoothed_images[co[0], co[1], 1] = g
                smoothed_images[co[0], co[1], 2] = b

        segments = slic(img_np, n_segments=625,
            compactness=10,
            max_num_iter=3,
            convert2lab=True,
            enforce_connectivity=False,
            slic_zero=False)
        
        regions = regionprops_table(segments, intensity_image=img_np, properties=('label','intensity_mean',
                                                                                'coords'))#, polarize])

        smoothed_images = np.zeros_like(img_np)
        for coord, r, g, b in zip(regions['coords'], regions['intensity_mean-0'], regions['intensity_mean-1'], regions['intensity_mean-2']):
            for co in coord:
                smoothed_images[co[0], co[1], 0] = r
                smoothed_images[co[0], co[1], 1] = g
                smoothed_images[co[0], co[1], 2] = b


        mask = c_img/255.
        mask[mask<=0.5] = 0
        mask[mask>0.5] = 1
        

        seq_mask = np.zeros([max(regions['label'])])
        
        
        for ind, coord in zip(regions['label'], regions['coords']):
            seq_mask[ind-1] = np.sum(mask[coord[:, 0], coord[:, 1]])/len(coord[:, 0])

        plt_image = seq_mask[segments-1].reshape([img_np.shape[0], img_np.shape[1]])
        plt_image = np.ravel(plt_image)
            

        mask = c_img.reshape(1, -1)/255.

        prec_d, recall_d = np.zeros([1, 256]), np.zeros([1, 256])
        prec_e, recall_e = np.zeros([1, 256]), np.zeros([1, 256])
        prec_max, recall_max = np.zeros([1, 256]), np.zeros([1, 256])
        pred_d = d_img.reshape(1, -1)/255.
        pred_e = e_img.reshape(1, -1)/255.
        
        thlist = np.linspace(0, 1 - 1e-10, 256)
        for j in range(256):
            y_temp_d = (pred_d >= thlist[j])
            y_temp_e = (pred_e >= thlist[j])
            y_temp_max = (plt_image >= thlist[j])
            tp_d = (y_temp_d * mask).sum(axis=-1)
            tp_e = (y_temp_e * mask).sum(axis=-1)
            tp_max = (y_temp_max * mask).sum(axis=-1)
            # avoid prec becomes 0
            prec_d[:, j], recall_d[:, j] = (tp_d + 1e-10) / (y_temp_d.sum(axis=-1) + 1e-10), (tp_d + 1e-10) / (mask.sum(axis=-1) + 1e-10)
            prec_e[:, j], recall_e[:, j] = (tp_e + 1e-10) / (y_temp_e.sum(axis=-1) + 1e-10), (tp_e + 1e-10) / (mask.sum(axis=-1) + 1e-10)
            prec_max[:, j], recall_max[:, j] = (tp_max + 1e-10) / (y_temp_max.sum(axis=-1) + 1e-10), (tp_max + 1e-10) / (mask.sum(axis=-1) + 1e-10)

        precs_d.append(prec_d)
        precs_e.append(prec_e)
        precs_max.append(prec_max)
        recalls_d.append(recall_d)
        recalls_e.append(recall_e)
        recalls_max.append(recall_max)
        


        prec_d = np.concatenate(precs_d, axis=0).mean(axis=0)
        prec_e = np.concatenate(precs_e, axis=0).mean(axis=0)
        prec_max = np.concatenate(precs_max, axis=0).mean(axis=0)
        recall_d = np.concatenate(recalls_d, axis=0).mean(axis=0)
        recall_e = np.concatenate(recalls_e, axis=0).mean(axis=0)
        recall_max = np.concatenate(recalls_max, axis=0).mean(axis=0)
        beta_square = 0.3
        
        f_score_d = (1 + beta_square) * prec_d * recall_d / (beta_square * prec_d + recall_d)
        f_score_e = (1 + beta_square) * prec_e * recall_e / (beta_square * prec_e + recall_e)
        f_score_max = (1 + beta_square) * prec_max * recall_max / (beta_square * prec_max + recall_max)

        f_scores_d.append(np.max(f_score_d))
        f_scores_e.append(np.max(f_score_e))
        f_scores_max.append(np.max(f_score_max))

        fig, ax = plt.subplots(nrows=2, ncols=3, clear=True, figsize=(15, 10))
        fig.subplots_adjust(wspace=0, hspace=0)
        ax[0, 0].imshow(a_img[...,::-1]/255.)
        ax[0, 0].axis('off')
        ax[0, 0].set_title('Image')
        ax[0, 1].imshow(mark_boundaries(smoothed_images, segments_256))
        ax[0, 1].axis('off')
        ax[0, 1].set_title('Superpixels')
        ax[0, 2].imshow(c_img, cmap='gray')
        ax[0, 2].axis('off')
        ax[0, 2].set_title('Ground Truth')
        ax[1, 0].imshow(d_img, cmap='gray')
        ax[1, 0].axis('off')
        ax[1, 0].set_title('TRACER')
        ax[1, 1].imshow(e_img, cmap='gray')
        ax[1, 1].axis('off')
        ax[1, 1].set_title('SuperFormer (Ours)')
        ax[1, 2].plot(f_scores_max, label='SP Upperbound')
        ax[1, 2].plot(f_scores_e, label='SF (Ours)')
        ax[1, 2].plot(f_scores_d, label='TRACER')
        ax[1, 2].set_ylabel('F1-score')
        ax[1, 2].set_xlabel('Frame index')
        ax[1, 2].legend()
        fig.tight_layout(h_pad=5, w_pad=5)

        if not os.path.exists('/home/eddie/waterloo/supertransformer/demo/'+object.split('/')[-1]):
            os.makedirs('/home/eddie/waterloo/supertransformer/demo/'+object.split('/')[-1])
        fig.savefig(os.path.join('/home/eddie/waterloo/supertransformer/demo/'+object.split('/')[-1], c.split('.')[0]+'jpg'))
        plt.close('all')
    
