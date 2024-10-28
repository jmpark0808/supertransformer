import matplotlib.pyplot as plt
import numpy as np



data = {'SAMNet': {'Params': 1.33, 'FLOPs': 0.5, 'MAE': 0.058, 'F1': 0.835, 'Marker': 'x', 'colour': 'r'},
        'HVPNet': {'Params': 1.23, 'FLOPs': 1.1, 'MAE': 0.058, 'F1': 0.839, 'Marker': 'x', 'colour': 'r'},
        'CorrNet': {'Params': 4.09, 'FLOPs': 21.1, 'MAE': 0.0466, 'F1': 0.847, 'Marker': 'x', 'colour': 'r'},
        'SeaNet': {'Params': 2.76, 'FLOPs': 1.7, 'MAE': 0.045, 'F1': 0.854, 'Marker': 'x', 'colour': 'r'},
        'MEANet': {'Params': 3.27, 'FLOPs': 9.62, 'MAE': 0.0454, 'F1': 0.863, 'Marker': 'x', 'colour': 'r'},
        'MSHNet': {'Params': 4.07, 'FLOPs': 6.11, 'MAE': 0.1251, 'F1': 0.7046, 'Marker': 'x', 'colour': 'r'},
        'Ours (XS)': {'Params': 1.12, 'FLOPs': 0.46, 'MAE': 0.053, 'F1': 0.847, 'Marker': 'o', 'colour': 'g'},
         'Ours (S)': {'Params': 2.18, 'FLOPs': 0.87, 'MAE': 0.045, 'F1': 0.866, 'Marker': 'o', 'colour': 'g'} }



fig, ax = plt.subplots(2, 2, figsize=(10, 10))

ax[0, 0].set_xlabel('Model Params. (M)')
ax[0, 0].set_ylabel('F1-score')

ax[0, 1].set_xlabel('GFLOPs')
ax[0, 1].set_ylabel('F1-score')

ax[1, 0].set_xlabel('Model Params. (M)')
ax[1, 0].set_ylabel('MAE')

ax[1, 1].set_xlabel('GFLOPs')
ax[1, 1].set_ylabel('MAE')

for net, value in data.items():
    ax[0, 0].scatter(value['Params'], value['F1'], marker=value['Marker'], color=value['colour'])
    ax[1, 0].scatter(value['Params'], value['MAE'], marker=value['Marker'], color=value['colour'])
    ax[0, 1].scatter(value['FLOPs'], value['F1'], marker=value['Marker'], color=value['colour'])
    ax[1, 1].scatter(value['FLOPs'], value['MAE'], marker=value['Marker'], color=value['colour'])

ax[0,0].plot([1.12, 2.18], [0.847, 0.866], color='g')
ax[0,1].plot([0.46, 0.87], [0.847, 0.866], color='g')
ax[1,0].plot([1.12, 2.18], [0.053, 0.045], color='g')
ax[1,1].plot([0.46, 0.87], [0.053, 0.045], color='g')

ax[0,1].set_xscale('log')
ax[1,1].set_xscale('log')
plt.show()

