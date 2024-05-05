import os
import numpy as np
from PIL import Image

annotation_path = '/mnt/dragon/Datasets/VideoSaliency/SegTrackv2/GroundTruth/'

selected_folder = ['bmx', 'cheetah', 'drift', 'hummingbird', 'monkeydog', 'penguin']
for folder in os.listdir(annotation_path):
    if folder in selected_folder:
        print(folder)
        object_ids = sorted(os.listdir(os.path.join(annotation_path, folder)))
        print(object_ids)
        all_paths = [sorted(os.listdir(os.path.join(annotation_path, folder, id)))for id in object_ids]
        for batch in zip(*all_paths):
            print(batch)
            img = np.array(Image.open(os.path.join(annotation_path, folder, '1', batch[0])).convert('L'))
            empty_image = np.zeros(img.shape)
            for idx, file_path in enumerate(batch):
                img = np.array(Image.open(os.path.join(annotation_path, folder, str(idx+1), file_path)).convert('L'))
                empty_image += img
            im = Image.fromarray(empty_image).convert('L')
            im.save(os.path.join(annotation_path, folder, file_path))
                
        
    else:
        continue