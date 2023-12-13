import numpy as np
import os
import matplotlib.pyplot as plt
import torch
from Models.SP_GAT_PyG import SP_GAT_PyG

seed = 82
spg_path = f'/mnt/dragon/gat_logs/{seed}/SPGFFT'
spgi_path = f'/mnt/dragon/gat_logs/{seed}/SPGIFFT'


spg_weights_path = f'/mnt/dragon/gat_logs/{seed}/SPGFFT'
spgi_weights_path = f'/mnt/dragon/gat_logs/{seed}/SPGIFFT'

spg_list = np.array(sorted([os.path.join(spg_path, f) for f in os.listdir(spg_path)]))
spgi_list = np.array(sorted([os.path.join(spgi_path, f) for f in os.listdir(spgi_path)]))

spg_weights_list = np.array(sorted([os.path.join(spg_weights_path, f) for f in os.listdir(spg_weights_path)]))
spgi_weights_list = np.array(sorted([os.path.join(spgi_weights_path, f) for f in os.listdir(spgi_weights_path)]))


fig, ax = plt.subplots(2, 3)
output_sum = 0
seq_mask_sum = 0
ei_sum = 0
x_sum = 0 
for spg, spgi in zip(spg_list, spgi_list):
    if 'output' in spg:
        spg_output = np.load(spg)
        spgi_output = np.load(spgi)
        diff = np.sum(np.abs(spg_output-spgi_output))
        output_sum += diff
        x = int(spg.split('.')[0].split('_')[-1])
        ax[0,0].scatter(x, diff, c='red')
        
    elif 'seq_mask' in spg:
        spg_seq_mask = np.load(spg)
        spgi_seq_mask = np.load(spgi)
        diff = np.sum(np.abs(spg_seq_mask-spgi_seq_mask))
        seq_mask_sum += diff
        x = int(spg.split('.')[0].split('_')[-1])

        ax[0,1].scatter(x, diff, c='blue')
        
    elif 'features_ei' in spg:
        spg_ei = np.load(spg)
        spgi_ei = np.load(spgi)
        diff = np.sum(np.abs(spg_ei-spgi_ei))
        ei_sum += diff
        x = int(spg.split('.')[0].split('_')[-1])
        ax[1,0].scatter(x, diff, c='green')
        
    elif 'features_x' in spg:
        spg_x = np.load(spg)
        spgi_x = np.load(spgi)
        diff = np.sum(np.abs(spg_x-spgi_x))
        x_sum += diff
        x = int(spg.split('.')[0].split('_')[-1])
        ax[1,1].scatter(x, diff, c='black')

weight_sum = 0
for spg, spgi in zip(spg_weights_list, spgi_weights_list):
    if 'model_weight' in spg:
        spg_model = SP_GAT_PyG(26, 16, None, 0, 8, 6, 625)
        spg_model.load_state_dict(torch.load(spg))
        spgi_model = SP_GAT_PyG(26, 16, None, 0, 8, 6, 625)
        spgi_model.load_state_dict(torch.load(spgi))
        for param_spg, param_spgi in zip(spg_model.parameters(), spgi_model.parameters()):
            diff = np.sum(np.abs(param_spg.detach().numpy()-param_spgi.detach().numpy()))

        weight_sum += diff
        x = int(spg.split('.')[0].split('_')[-1])
        ax[0,2].scatter(x, diff, c='purple')
        
    
        

ax[0,0].set_xlabel('Batch idx')
ax[0,0].set_ylabel('Difference')
ax[0,0].set_title(f'Model output, diff sum {output_sum}')

ax[0,1].set_xlabel('Batch idx')
ax[0,1].set_ylabel('Difference')
ax[0,1].set_title(f'Label, diff sum {seq_mask_sum}')


ax[1,0].set_xlabel('Batch idx')
ax[1,0].set_ylabel('Difference')
ax[1,0].set_title(f'Edge index, diff sum {ei_sum}')


ax[1,1].set_xlabel('Batch idx')
ax[1,1].set_ylabel('Difference')
ax[1,1].set_title(f'Input, diff sum {x_sum}')

ax[0,2].set_xlabel('Batch idx')
ax[0,2].set_ylabel('Difference')
ax[0,2].set_title(f'Weights, diff sum {x_sum}')
plt.show()
