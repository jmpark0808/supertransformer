
import os
import xml.etree.ElementTree as ET
import shutil


root_dir = '/home/eddie/Datasets/ImageNet'

image_list = sorted(os.listdir('{}/val'.format(root_dir)))
target_list = sorted(os.listdir('{}/xml'.format(root_dir)))





for item in range(len(image_list)):
    img_name = '{}/val/{}'.format(root_dir, image_list[item])
    target_name = '{}/xml/{}'.format(root_dir, target_list[item]) 
    print(target_name)
    target = ET.parse(target_name)
    root = target.getroot()
    target = root[5][0].text
    
    target_folder = '{}/val/{}'.format(root_dir, target)
    
    if os.path.exists(target_folder):
        shutil.copyfile(img_name, os.path.join(target_folder, image_list[item]))
    else:
        os.makedirs(target_folder)
        shutil.copyfile(img_name, os.path.join(target_folder, image_list[item]))
    