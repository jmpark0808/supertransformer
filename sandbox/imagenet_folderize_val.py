
import os
import xml.etree.ElementTree as ET
import shutil


root_dir = '/mnt/dragon/Datasets/imagenet-object-localization-challenge/ILSVRC/'

image_list = sorted(os.listdir('{}/Data/CLS-LOC/val'.format(root_dir)))
target_list = sorted(os.listdir('{}/Annotations/CLS-LOC/val'.format(root_dir)))





for item in range(len(image_list)):
    img_name = '{}/Data/CLS-LOC/val/{}'.format(root_dir, image_list[item])
    target_name = '{}/Annotations/CLS-LOC/val/{}'.format(root_dir, target_list[item]) 
   
    target = ET.parse(target_name)
    root = target.getroot()
    target = root[5][0].text
    
    target_folder = '{}/Data/CLS-LOC/val/{}'.format(root_dir, target)
    
    if os.path.exists(target_folder):
        shutil.copyfile(img_name, os.path.join(target_folder, image_list[item]))
    else:
        os.makedirs(target_folder)
        shutil.copyfile(img_name, os.path.join(target_folder, image_list[item]))
    