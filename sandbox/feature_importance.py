

import os
from Models.SP_TFM import SP_TFM_REL_test
import torch
from dataset.superpixel_fast import SPDataset
from captum.attr import (
    DeepLift
)
from torch.utils.data import DataLoader
import numpy as np
import tqdm


weights = '/mnt/hdd/Experiments/temp/1204-142701_42923667/epoch=199-step=112200.ckpt'
model = SP_TFM_REL_test(26, 1, 16, 8, 6, 0).cuda()

checkpoint = torch.load(weights)
for key in list(checkpoint['state_dict'].keys()):
    checkpoint['state_dict'][key.replace('supert.', '')] = checkpoint['state_dict'].pop(key)

model.load_state_dict(checkpoint['state_dict'])

test_dir = '/mnt/hdd/Datasets/DUTS/TE'
test_image_list = sorted([os.path.join(os.path.join(test_dir, 'Image'), f) for f in os.listdir(os.path.join(test_dir, 'Image'))])
test_mask_list = sorted([os.path.join(os.path.join(test_dir, 'Mask'), f) for f in os.listdir(os.path.join(test_dir, 'Mask'))])

data_test = SPDataset(test_image_list, test_mask_list, 625, 256,  10, 'SPF', False,
                                 10, False, False, None, None,  1)

loader = DataLoader(data_test, 1, False, num_workers=12)
model.eval()

all_attributions = []
all_deltas = []

for batch in tqdm.tqdm(loader):
    features = batch['features'].cuda()
    with torch.no_grad():
        out = model(features)

        baseline = torch.zeros_like(features).cuda()

        dl = DeepLift(model)
        attributions, delta = dl.attribute(features, baseline, target=0, return_convergence_delta=True)
        
        summed_attributions = torch.sum(torch.abs(attributions), dim=(1))
        abs_delta = torch.abs(delta)

        dl = DeepLift(model)
        attributions, delta = dl.attribute(features, baseline, target=1, return_convergence_delta=True)
        summed_attributions +=  torch.sum(torch.abs(attributions), dim=(1))
        all_attributions.append(summed_attributions.detach().cpu().numpy())
        abs_delta = (abs_delta+torch.abs(delta))/2.
        all_deltas.append(abs_delta)


attributions = np.mean(np.stack(all_attributions, axis=0), axis=0)
delta = np.mean(np.stack(all_deltas, axis=0), axis=0)

print(attributions.shape)
print(delta.shape)

import matplotlib.pyplot as plt
plt.plot(attributions)
plt.ylabel('Importance')
plt.xlabel('Feature Index')
scene_var = ['Pos_x', 'Pos_y', 'Mean R', 'Mean G', 'Mean B', 'Var R', 'Var G', 'Var B', '', '',
              '', '', '', '', '', '', '', 'FFT', '', '', '', '', '', '', '', '', '', '']
plt.xticks(list(range(len(scene_var))), scene_var, rotation=90)
plt.title(f'Delta {delta}')
plt.show()