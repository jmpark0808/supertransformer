# load image 
from torchvision import transforms, datasets
import torch
from PIL import Image
from Blocks.blocks import SLICPyTorch
data_dir = '/mnt/hdd/Datasets/DUTS/TR/Image/ILSVRC2012_test_00000004.jpg'
img = Image.open(data_dir)

