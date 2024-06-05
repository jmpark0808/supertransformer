import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


df = pd.read_csv('/home/eddie/Downloads/ds_vs_sp.csv')
xs = df['FLOPS']
labels = df['Datatype']
f1s = df['F1-score']
min_f1 = np.min(f1s)
max_f1 = np.max(f1s)
resolution = df['Resolution']
all_resolutions = [100, 225, 400, 900, 2500]
fig, axs = plt.subplots(1, 5)
for idx, r in enumerate(all_resolutions):
    temp_index = resolution == r
    temp_labels = labels[temp_index]
    temp_xs = xs[temp_index]
    temp_f1s = f1s[temp_index]

    ds_index = temp_labels == 'DS'
    sp_index = temp_labels == 'SP'
    
    ds_xs = temp_xs[ds_index]
    ds_f1 = temp_f1s[ds_index]
    ds_f1 = [x for _, x in sorted(zip(ds_xs, ds_f1))]
    ds_xs = sorted(ds_xs)
    sp_xs = temp_xs[sp_index]
    sp_f1 = temp_f1s[sp_index]
    sp_f1 = [x for _, x in sorted(zip(sp_xs, sp_f1))]
    sp_xs = sorted(sp_xs)
    axs[idx].plot(ds_xs, ds_f1, label='Downsample', c='red')
    axs[idx].scatter(ds_xs, ds_f1, c='red')
    axs[idx].plot(sp_xs, sp_f1, label='Superpixels', c='g')
    axs[idx].scatter(sp_xs, sp_f1, c='g')
    axs[idx].legend()
    axs[idx].set_ylim(min_f1-0.05, max_f1+0.05)
    axs[idx].set_title(f'Number of pixels/superpixels: {r}')
fig.supxlabel('FLOPS')
fig.supylabel('F1-score')
plt.show()