import os
import cv2
import matplotlib.pyplot as plt


images_dir = '/mnt/hdd/Datasets/DUTS/DUTS-TE/Image'
mask_dir = '/mnt/hdd/Datasets/DUTS/DUTS-TE/Mask'
gat_dir = '/home/eddie/waterloo/supertransformer/visualization/SP_GAT/DUTS-TE'
sc_dir = '/home/eddie/waterloo/supertransformer/visualization/SP_CNN_LIN/DUTS-TE'
gf_dir = '/home/eddie/waterloo/supertransformer/visualization/SP_Baseline_LAP/DUTS-TE'
sf_dir = '/home/eddie/waterloo/supertransformer/visualization/SP_TFM/DUTS-TE'
mbnet_dir = '/home/eddie/waterloo/TRACER/mask_MobileNet/DUTS/Test/images'
vst_dir = '/home/eddie/waterloo/VST/RGB_VST/preds_DUTS-TE/DUTS/RGB_VST/'
tracer_dir = '/home/eddie/waterloo/TRACER/mask/DUTS/Test/images/'


file_names = [f.split('.')[0] for f in os.listdir(images_dir)]
dim = 128

for file in file_names:
    gat_pred = cv2.resize(cv2.imread(os.path.join(gat_dir, file+'.jpg')), (dim,dim))
    sc_pred = cv2.resize(cv2.imread(os.path.join(sc_dir, file+'.jpg')), (dim,dim))
    gf_pred = cv2.resize(cv2.imread(os.path.join(gf_dir, file+'.jpg')), (dim,dim))
    sf_pred = cv2.resize(cv2.imread(os.path.join(sf_dir, file+'.jpg')), (dim,dim))
    mbnet_pred = cv2.resize(cv2.imread(os.path.join(mbnet_dir, file+'.png')), (dim,dim))
    vst_pred = cv2.resize(cv2.imread(os.path.join(vst_dir, file+'.png')), (dim,dim))
    tracer_pred = cv2.resize(cv2.imread(os.path.join(tracer_dir, file+'.png')), (dim,dim))
    raw_image = cv2.resize(cv2.imread(os.path.join(images_dir, file+'.jpg')), (dim,dim))
    mask = cv2.resize(cv2.imread(os.path.join(mask_dir, file+'.png')), (dim,dim))
    
    fig, ax = plt.subplots(nrows=1, ncols=9)
    ax[0].imshow(raw_image[...,::-1])
    ax[1].imshow(mask[...,::-1])
    ax[2].imshow(sf_pred[...,::-1])
    ax[3].imshow(gat_pred[...,::-1])
    ax[4].imshow(sc_pred[...,::-1])
    ax[5].imshow(gf_pred[...,::-1])
    ax[6].imshow(vst_pred[...,::-1])
    ax[7].imshow(mbnet_pred[...,::-1])
    ax[8].imshow(tracer_pred[...,::-1])
    [axi.set_axis_off() for axi in ax.ravel()]
    plt.show()

