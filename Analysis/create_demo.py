import matplotlib.pyplot as plt
import os
import cv2
import numpy as np

monkey_dir = '/mnt/hdd/Datasets/SegTrackv2/JPEGImages/monkey'
girl_dir = '/mnt/hdd/Datasets/SegTrackv2/JPEGImages/girl'
soldier_dir = '/mnt/hdd/Datasets/SegTrackv2/JPEGImages/soldier'
bird_dir = '/mnt/hdd/Datasets/SegTrackv2/JPEGImages/bird_of_paradise'

monkey_gt_dir = '/mnt/hdd/Datasets/SegTrackv2/GroundTruth/monkey'
girl_gt_dir = '/mnt/hdd/Datasets/SegTrackv2/GroundTruth/girl'
soldier_gt_dir = '/mnt/hdd/Datasets/SegTrackv2/GroundTruth/soldier'
bird_gt_dir = '/mnt/hdd/Datasets/SegTrackv2/GroundTruth/bird_of_paradise'

monkey_tracer_dir = '/home/abcd/abcde/TRACER/mask/monkey'
girl_tracer_dir = '/home/abcd/abcde/TRACER/mask/girl'
soldier_tracer_dir = '/home/abcd/abcde/TRACER/mask/soldier'
bird_tracer_dir = '/home/abcd/abcde/TRACER/mask/bird_of_paradise'

monkey_sf_dir = '/home/abcd/abcde/supertransformer/visualization/SP_TFM/Monkey'
girl_sf_dir = '/home/abcd/abcde/supertransformer/visualization/SP_TFM/Girl'
soldier_sf_dir = '/home/abcd/abcde/supertransformer/visualization/SP_TFM/Solider'
bird_sf_dir = '/home/abcd/abcde/supertransformer/visualization/SP_TFM/Bird_of_paradise'

all_objects = [monkey_dir, girl_dir, soldier_dir, bird_dir]
ground_truths = [monkey_gt_dir, girl_gt_dir, soldier_gt_dir, bird_gt_dir]
tracers = [monkey_tracer_dir, girl_tracer_dir, soldier_tracer_dir, bird_tracer_dir]
sfs = [monkey_sf_dir, girl_sf_dir, soldier_sf_dir, bird_sf_dir]

for object, gt, tracer, sf in zip(all_objects, ground_truths, tracers, sfs):
    objects = sorted([x for x in os.listdir(object) if not 'smooth' in x])
    objects_sp =  sorted([x for x in os.listdir(object) if 'smooth' in x])
    gts = sorted(os.listdir(gt))
    tr = sorted(os.listdir(tracer))
    s = sorted(os.listdir(sf))

    for a, b, c, d, e in zip(objects, objects_sp, gts, tr, s):
        a_img = cv2.imread(os.path.join(object, a))
        
        b_img = cv2.imread(os.path.join(object, b))
        c_img = cv2.imread(os.path.join(gt, c))
        d_img = cv2.imread(os.path.join(tracer, d))
        e_img = cv2.imread(os.path.join(sf, e))

        a_img = cv2.resize(a_img, (224, 224))
        b_img = cv2.resize(b_img, (224, 224))
        c_img = cv2.resize(c_img, (224, 224))
        d_img = cv2.resize(d_img, (224, 224))
        e_img = cv2.resize(e_img, (224, 224))

        fig, ax = plt.subplots(nrows=1, ncols=5, clear=True, figsize=(15, 5))
        fig.subplots_adjust(wspace=0, hspace=0)
        ax[0].imshow(a_img[...,::-1]/255.)
        ax[0].axis('off')
        ax[0].set_title('Image')
        ax[1].imshow(b_img[...,::-1]/255.)
        ax[1].axis('off')
        ax[1].set_title('Superpixels')
        ax[2].imshow(c_img)
        ax[2].axis('off')
        ax[2].set_title('Ground Truth')
        ax[3].imshow(d_img)
        ax[3].axis('off')
        ax[3].set_title('TRACER')
        ax[4].imshow(e_img)
        ax[4].axis('off')
        ax[4].set_title('SuperFormer (Ours)')

        if not os.path.exists('/home/abcd/abcde/supertransformer/demo/'+object.split('/')[-1]):
            os.makedirs('/home/abcd/abcde/supertransformer/demo/'+object.split('/')[-1])
        fig.savefig(os.path.join('/home/abcd/abcde/supertransformer/demo/'+object.split('/')[-1], c.split('.')[0]+'jpg'))
        plt.close('all')
    
