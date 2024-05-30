import pickle
import matplotlib.pyplot as plt
import numpy as np

with open('/home/eddie/waterloo/supertransformer/Analysis/segments_plot_data.pkl', 'rb') as f:
    loaded_dict = pickle.load(f)
plt.figure(figsize=(10,10))
segment_numbers = loaded_dict['segment_numbers']

compactness = [0.1, 1, 10, 50]
for compact in compactness:
    all_ious = loaded_dict[compact]
    plt.plot(np.power(segment_numbers, 2), all_ious, label=f'SP, C {compact}')
    plt.scatter(np.power(segment_numbers, 2), all_ious)
    if compact == 10:
        for i, j in zip(segment_numbers, all_ious):
            if i <= 10000:
                plt.text(i**2, j+0.005, '{}'.format(i**2))
            else:
                plt.text(i**2, j+0.002, '{}'.format(i**2))

fs = 30

with open('/home/eddie/waterloo/supertransformer/Analysis/resizing_plot_data_updated.pkl', 'rb') as f:
    loaded_dict = pickle.load(f)

segment_numbers = loaded_dict['image_resolutions']


all_ious = loaded_dict['ious']
plt.plot(np.power(segment_numbers, 2), all_ious, label='Downsample')
plt.scatter(np.power(segment_numbers, 2), all_ious)

for i, j in zip(segment_numbers, all_ious):
    if i <= 10000:
        plt.text(i**2, j+0.005, '{}'.format(i**2))
    else:
        plt.text(i**2, j+0.002, '{}'.format(i**2))

plt.title(f'Maximum F1-score accuracy For a Given \n  Number of Pixels/Superpixels', fontsize=fs)
plt.xlabel('Number of Pixels/Superpixels (log scale)', fontsize=fs)
plt.ylabel('F1-score', fontsize=fs)
plt.xscale('log')
plt.xticks(fontsize=fs, rotation=45)
plt.yticks(fontsize=fs)
plt.legend(loc="lower right", fontsize=fs, title_fontsize=fs)
plt.vlines(x=np.power(segment_numbers, 2), ymin=0.85, ymax=1, ls=":")
plt.tight_layout()
# plt.savefig(f'compactness.jpg', bbox_inches='tight')
plt.show()
    