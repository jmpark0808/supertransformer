



import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def mIoU(y_true, y_pred):
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    intersection = np.sum(y_true*y_pred)
    score = (intersection + 1.)/(np.sum(y_true) + np.sum(y_pred) - intersection + 1.)
    return score

def f1score(y_pred, y_true):
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    
    y_temp = (y_pred >= 0.5).astype(np.float32)
    tp = np.sum(y_temp*y_true)
    
    
    # avoid prec becomes 0
    prec, recall = (tp + 1e-10) / (np.sum(y_temp) + 1e-10), (tp + 1e-10) / (np.sum(y_true) + 1e-10)
    
    beta_square = 0.3
    f_score = (1 + beta_square) * prec * recall / (beta_square * prec + recall)
    return f_score

mask_dir = '/home/eddie/Datasets/DUTS/DUTS-TE/Mask/'
img_dir = '/home/eddie/Datasets/DUTS/DUTS-TE/Image/'
# samnet_dir = '/home/eddie/Qualitative/SAMNet/DUTS-TE/DUTS-TE/'
# hvpnet_dir = '/home/eddie/Qualitative/HVPNet/DUTS-TE/'
corrnet_dir = '/home/eddie/Qualitative/CorrNet/DUTS-TE/'
seanet_dir = '/home/eddie/Qualitative/SeaNet/DUTS-TE/'
meanet_dir = '/home/eddie/Qualitative/MEANet/DUTS-TE/'
mshnet_dir = '/home/eddie/Qualitative/MSHNet/DUTS-TE/'
inas_dir = '/home/eddie/Qualitative/iNas/DUTS-TE/'
isaanet_dir =  '/home/eddie/Qualitative/ISAANet/DUTS-TE/'
sf_dir = '/home/eddie/Qualitative/SF-S/DUTS-TE/'

# samnet_file_names = os.listdir(samnet_dir)
# hvpnet_file_names = os.listdir(hvpnet_dir)
corrnet_file_names = os.listdir(corrnet_dir)
seanet_file_names = os.listdir(seanet_dir)
meanet_file_names = os.listdir(meanet_dir)
mshnet_file_names = os.listdir(mshnet_dir)
inas_file_names = os.listdir(inas_dir)
isaanet_file_names = os.listdir(mshnet_dir)
sfnet_file_names = os.listdir(sf_dir)


skip_files = ['ILSVRC2012_test_00028731.png', 'sun_bivvtruztkqmtpnf.png',
               'ILSVRC2012_test_00043101.png', 'ILSVRC2012_test_00000606.png',
               'ILSVRC2012_test_00078616.png', 'ILSVRC2012_test_00085854.png',
               'ILSVRC2013_test_00008577.png','ILSVRC2012_test_00002350.png',
               'ILSVRC2013_test_00004480.png']

num_rows = 7

fig, ax = plt.subplots(num_rows, 8, figsize = (8*2,num_rows*2))
ind = 0
for file in os.listdir(mask_dir):
    if ind == num_rows:
        break
    if file.endswith('jpg'):
        other_format = file.replace('.jpg', '.png')
    elif file.endswith('png'):
        other_format = file.replace('.png', '.jpg')
    else:
        assert 0, 'Unrecognized format'

    try:
        img = Image.open(os.path.join(img_dir, file))
    except:
        img = Image.open(os.path.join(img_dir, other_format))

    img = np.array(img.resize((300, 300)))/255.
    mask = Image.open(os.path.join(mask_dir, file)).convert('L')
    mask = np.array(mask.resize((300, 300)))/255.
    
    # try:
    #     samnet_img = Image.open(os.path.join(samnet_dir, file)).convert('L')
    # except:
    #     samnet_img = Image.open(os.path.join(samnet_dir, other_format)).convert('L')
    # samnet_img = np.array(samnet_img.resize((300, 300)))/255.

    # try:
    #     hvpnet_img = Image.open(os.path.join(hvpnet_dir, file)).convert('L')
    # except:
    #     hvpnet_img = Image.open(os.path.join(hvpnet_dir, other_format)).convert('L')
    # hvpnet_img = np.array(hvpnet_img.resize((300, 300)))/255.

    try:
        corrnet_img = Image.open(os.path.join(corrnet_dir, file)).convert('L')
    except:
        corrnet_img = Image.open(os.path.join(corrnet_dir, other_format)).convert('L')
    corrnet_img = np.array(corrnet_img.resize((300, 300)))/255.
    try:
        seanet_img = Image.open(os.path.join(seanet_dir, file)).convert('L')
    except:
        seanet_img = Image.open(os.path.join(seanet_dir, other_format)).convert('L')
    seanet_img = np.array(seanet_img.resize((300, 300)))/255.
    try:
        meanet_img = Image.open(os.path.join(meanet_dir, file)).convert('L')
    except:
        meanet_img = Image.open(os.path.join(meanet_dir, other_format)).convert('L')
    meanet_img = np.array(meanet_img.resize((300, 300)))/255.
    try:
        mshnet_img = Image.open(os.path.join(mshnet_dir, file)).convert('L')
    except:
        mshnet_img = Image.open(os.path.join(mshnet_dir, other_format)).convert('L')
    mshnet_img = np.array(mshnet_img.resize((300, 300)))/255.

    try:
        inas_img = Image.open(os.path.join(inas_dir, file)).convert('L')
    except:
        inas_img = Image.open(os.path.join(inas_dir, other_format)).convert('L')
    inas_img = np.array(inas_img.resize((300, 300)))/255.

    try:
        isaanet_img = Image.open(os.path.join(isaanet_dir, file)).convert('L')
    except:
        isaanet_img = Image.open(os.path.join(isaanet_dir, other_format)).convert('L')
    isaanet_img = np.array(isaanet_img.resize((300, 300)))/255.
    try:
        sf_img = Image.open(os.path.join(sf_dir, file)).convert('L')
    except:
        sf_img = Image.open(os.path.join(sf_dir, other_format)).convert('L')
    sf_img = np.array(sf_img.resize((300, 300)))/255.

    
    # samnet_mae = f1score(samnet_img,mask)
    # hvpnet_mae =  f1score(hvpnet_img,mask)
    corrnet_mae =  f1score(corrnet_img,mask)
    seanet_mae =  f1score(seanet_img,mask)
    meanet_mae = f1score(meanet_img,mask)
    mshnet_mae =  f1score(mshnet_img,mask)
    inas_mae = f1score(inas_img,mask)
    isaanet_mae =  f1score(isaanet_img,mask)
    sf_mae =  f1score(sf_img,mask)

    if np.max(np.array([ corrnet_mae, seanet_mae, meanet_mae, mshnet_mae, inas_mae, sf_mae, isaanet_mae])) == sf_mae \
        and sf_mae > 0.95 and np.mean(mask) < 0.3 and file not in skip_files:
        ax[ind, 0].imshow(img)
        ax[ind, 1].imshow(mask, cmap='gray')
        # ax[ind, 2].imshow(samnet_img, cmap='gray')
        # ax[ind, 3].imshow(hvpnet_img, cmap='gray')
        ax[ind, 2].imshow(corrnet_img, cmap='gray')
        ax[ind, 3].imshow(seanet_img, cmap='gray')
        ax[ind, 4].imshow(meanet_img, cmap='gray')
        ax[ind, 5].imshow(mshnet_img, cmap='gray')
        ax[ind, 6].imshow(isaanet_img, cmap='gray')
        ax[ind, 7].imshow(sf_img, cmap='gray')
        
        print(file)
      


        ind += 1

for axs in ax.ravel():
    axs.set_axis_off()

fig.tight_layout()
fig.subplots_adjust(hspace=0.05, wspace=0.05)
fig.savefig('/mnt/hdd/Figures/SuperFormer/qualitative_v2.pdf', format='pdf')




    
    


   