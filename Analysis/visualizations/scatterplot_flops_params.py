import matplotlib.pyplot as plt
import numpy as np
from adjustText import adjust_text
import matplotlib

data = {'SAMNet': {'Params': 1.33, 'FLOPs': 0.5, 'MAE': 0.058, 'F1': 0.835, 'Marker': 'x', 'colour': 'r'},
        'HVPNet': {'Params': 1.23, 'FLOPs': 1.1, 'MAE': 0.058, 'F1': 0.839, 'Marker': 'x', 'colour': 'r'},
        'CorrNet': {'Params': 4.09, 'FLOPs': 21.1, 'MAE': 0.0466, 'F1': 0.847, 'Marker': 'x', 'colour': 'r'},
        'SeaNet': {'Params': 2.76, 'FLOPs': 1.7, 'MAE': 0.045, 'F1': 0.854, 'Marker': 'x', 'colour': 'r'},
        'MEANet': {'Params': 3.27, 'FLOPs': 9.62, 'MAE': 0.0454, 'F1': 0.863, 'Marker': 'x', 'colour': 'r'},
        'MSHNet': {'Params': 4.07, 'FLOPs': 6.11, 'MAE': 0.1251-0.06, 'F1': 0.7046+0.1175, 'Marker': 'x', 'colour': 'r'},
        'Ours (XS)': {'Params': 1.12, 'FLOPs': 0.46, 'MAE': 0.053, 'F1': 0.847, 'Marker': 'o', 'colour': 'g'},
         'Ours (S)': {'Params': 2.18, 'FLOPs': 0.87, 'MAE': 0.045, 'F1': 0.866, 'Marker': 'o', 'colour': 'g'} }


text1 = []
text2 = []
text3 = []
text4 = []
fig, ax = plt.subplots(1, 2, figsize=(10, 5))

label_fontsize = 15
y_spaces = np.linspace(0.82, 0.87, 11)
y_labels = [str(format(x, '.3f')) for x in y_spaces]
y_labels[0] = '0.700'
y_labels[1] = '0.705'
y_labels[2] = '...'
ax[ 0].set_xlabel('Model Params. (M)', fontsize=label_fontsize)
ax[ 0].set_ylabel('F1-score', fontsize=label_fontsize)
ax[ 0].set_yticks(y_spaces)
ax[ 0].set_ylim(0.82, 0.87)
ax[ 0].set_yticklabels(y_labels) 
ax[ 1].set_xlabel('GFLOPs', fontsize=label_fontsize)
ax[ 1].set_ylabel('F1-score', fontsize=label_fontsize)
ax[ 1].set_yticks(y_spaces)
ax[ 1].set_ylim(0.82, 0.87)
ax[ 1].set_yticklabels(y_labels) 
ax[1].set_xscale('log')
ax[1].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax[0].plot([1.12, 2.18], [0.847, 0.866], color='g')
ax[1].plot([0.46, 0.87], [0.847, 0.866], color='g')


for net, value in data.items():
    ax[ 0].scatter(value['Params'], value['F1'], marker=value['Marker'], color=value['colour'])
    ax[1].scatter(value['FLOPs'], value['F1'], marker=value['Marker'], color=value['colour'])
    text1.append(ax[ 0].text(value['Params'], value['F1'], net, fontsize=12))
    text3.append(ax[ 1].text(value['FLOPs'], value['F1'], net,  fontsize=12))

adjust_text(text1,   ax=ax[0], min_arrow_len=5) #arrowprops=dict(arrowstyle="-", color='b', lw=1),
adjust_text(text3,   ax=ax[1], min_arrow_len=5)

plt.tight_layout()
plt.savefig('/mnt/d/Figures/SuperFormer/scatter_f1.pdf', format='pdf')

fig, ax = plt.subplots(1, 2, figsize=(10, 5))

y_spaces = np.linspace(0.0425, 0.0675, 11)
y_labels = [str(format(x, '.4f')) for x in y_spaces]
y_labels[-1] = '0.1275'
y_labels[-2] = '0.1250'
y_labels[-3] = '...'
ax[ 0].set_xlabel('Model Params. (M)', fontsize=label_fontsize)
ax[ 0].set_ylabel('MAE', fontsize=label_fontsize)
ax[ 0].set_ylim(0.0425, 0.0675)
ax[ 0].set_yticks(y_spaces)
ax[ 0].set_yticklabels(y_labels) 
ax[ 1].set_xlabel('GFLOPs', fontsize=label_fontsize)
ax[ 1].set_ylabel('MAE', fontsize=label_fontsize)
ax[ 1].set_ylim(0.0425, 0.0675)
ax[ 1].set_yticks(y_spaces)
ax[ 1].set_yticklabels(y_labels) 

ax[1].set_xscale('log')

ax[1].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())


for net, value in data.items():
    ax[ 0].scatter(value['Params'], value['MAE'], marker=value['Marker'], color=value['colour'])
    ax[ 1].scatter(value['FLOPs'], value['MAE'], marker=value['Marker'], color=value['colour'])

    text2.append(ax[ 0].text(value['Params'], value['MAE'], net,  fontsize=12))
    text4.append(ax[ 1].text(value['FLOPs'], value['MAE'],net,   fontsize=12))
  



ax[0].plot([1.12, 2.18], [0.053, 0.045], color='g')
ax[1].plot([0.46, 0.87], [0.053, 0.045], color='g')


adjust_text(text2,    ax=ax[0], min_arrow_len=5)

adjust_text(text4,   ax=ax[1], min_arrow_len=5)


plt.tight_layout()
plt.savefig('/mnt/d/Figures/SuperFormer/scatter_mae.pdf', format='pdf')
