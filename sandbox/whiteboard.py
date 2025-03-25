from PIL import Image
from dataset.SLIC_to_SP import compute_superpixel_representation_batch
from torchvision.transforms import ToTensor
from Blocks.swinunet_mix_rope import SwinUTransformer
import torch
import time
import numpy as np
from skimage.segmentation import slic
from skimage.feature import local_binary_pattern
from skimage.measure import regionprops_table
img = Image.open('/home/eddie/Datasets/DUTS/DUTS-TE/Image/ILSVRC2012_test_00000003.jpg')
mask =  Image.open('/home/eddie/Datasets/DUTS/DUTS-TE/Mask/ILSVRC2012_test_00000003.png')

img = img.convert('RGB').resize((224, 224))
mask = mask.convert('L').resize((224, 224))

img_np = np.array(img)
img_gray = np.array(img.convert('L'))

def lbp(region, intensities):
    (hist, _) = np.histogram(intensities[region].ravel(),
            bins=np.arange(0, 8+3),
            range=(0, 8+2))
    hist = hist.astype("float")
    # hist /= (hist.sum() + 1e-7)
    return hist

segments = slic(img_np, n_segments=3136,
            compactness=10,
            max_num_iter=10,
            convert2lab=True,
            enforce_connectivity=False,
            slic_zero=False)

       
def image_stdev(region, intensities):
    # note the ddof arg to get the sample var if you so desire!
    return np.std(intensities[region])
        


                
lbp_np = local_binary_pattern(img_gray, 8, 1, method='uniform')


tt = ToTensor()
img_tensor = tt(img).cuda()
mask_tensor = tt(img).cuda()

model = SwinUTransformer(img_size=56, in_chans=24, patch_size=1, window_size=7,
                                       embed_dim=[32, 64, 128], depths=[2, 2, 6],
                                         num_heads=[2, 4, 8], mlp_ratio=2, attn_drop_rate=0, drop_rate=0,
                                         qkv_bias=False, drop_path_rate=0.1).cuda()


model.eval()
images = img_tensor.unsqueeze(0)
masks = mask_tensor.unsqueeze(0)
lbps = torch.tensor(lbp_np).unsqueeze(0).cuda()
segments = torch.tensor(segments).unsqueeze(0).cuda()
all_times = []

with torch.no_grad():
    for _ in range(1000):
        start = time.time()
        inp, _ = compute_superpixel_representation_batch(images, lbps, segments-1, None, 3136)
        inp = inp.reshape(1, 56, 56, 26).permute(0, 3, 1, 2)
        model(inp)
        end = time.time()
        all_times.append(end-start)

print(np.mean(all_times))



