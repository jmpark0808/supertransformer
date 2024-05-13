import sys
sys.path.insert(0, '/home/eddie/waterloo/supertransformer')
from dataset.imagenet_images import ImageNetDataModule
from dataset.imagenet_aug import SPImageNetAugDataModule
from tqdm import tqdm


dict_args = {'train_dir': '/mnt/dragon/Datasets/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/train',
             'test_dir': '/mnt/dragon/Datasets/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/val',
             'batch_size': 1, 'num_workers': 12, 'seed': 22, 'size': 224, 
             'num_seg':1024, 'coeff': 10, 'compactness': 10, 'ignore_phase': False}




images = ImageNetDataModule(**dict_args).train_dataloader()
slics = SPImageNetAugDataModule(**dict_args).train_dataloader()
stop_count = 1000
print('start images')
start = 0
for batch in tqdm(images):
    start += 1
    if start == stop_count:
        break

print('start slic')
start = 0
for batch in tqdm(slics):
    start += 1
    if start == stop_count:
        break

