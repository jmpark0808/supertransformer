import pickle
import matplotlib.pyplot as plt
import numpy as np

with open('/home/eddie/waterloo/supertransformer/segments_plot_data.pkl', 'rb') as f:
    loaded_dict = pickle.load(f)
# plt.figure(figsize=(10,10))
# segment_numbers = loaded_dict['segment_numbers']

# compactness = np.linspace(0.1, 2, 5)
# for compact in compactness:
#     all_ious = loaded_dict[compact]
#     plt.plot(np.power(segment_numbers, 2), all_ious, label=f'SP, C {compact}')
#     plt.scatter(np.power(segment_numbers, 2), all_ious)
#     if compact == 10:
#         for i, j in zip(segment_numbers, all_ious):
#             if i <= 10000:
#                 plt.text(i**2, j+0.005, '{}'.format(i**2))
#             else:
#                 plt.text(i**2, j+0.002, '{}'.format(i**2))


with open('/home/eddie/waterloo/supertransformer/segments_plot_data.pkl', 'rb') as f:
    loaded_dict = pickle.load(f)

segment_numbers = loaded_dict['segment_numbers']
fig, ax = plt.subplots(1, 2, figsize=(20, 10))
compactness = [0.1, 1, 10, 50]

for compact in compactness:
    all_ious, all_maes = loaded_dict[compact]
    ax[0].plot(np.power(segment_numbers, 2), all_ious, label=f'SP, C {compact}')
    ax[0].scatter(np.power(segment_numbers, 2), all_ious)
    if compact == 10:
        for i, j in zip(segment_numbers, all_ious):
            if i <= 10000:
                ax[0].text(i**2, j+0.005, '{}'.format(i**2))
            else:
                ax[0].text(i**2, j+0.002, '{}'.format(i**2))

    ax[1].plot(np.power(segment_numbers, 2), all_maes, label=f'SP, C {compact}')
    ax[1].scatter(np.power(segment_numbers, 2), all_maes)
    if compact == 10:
        for i, j in zip(segment_numbers, all_maes):
            if i <= 10000:
                ax[1].text(i**2, j+0.005, '{}'.format(i**2))
            else:
                ax[1].text(i**2, j+0.002, '{}'.format(i**2))

fs = 30

# with open('/home/eddie/waterloo/supertransformer/Analysis/resizing_plot_data_updated.pkl', 'rb') as f:
#     loaded_dict = pickle.load(f)

# segment_numbers = loaded_dict['image_resolutions']


# all_ious = loaded_dict['ious']
# plt.plot(np.power(segment_numbers, 2), all_ious, label='Downsample')
# plt.scatter(np.power(segment_numbers, 2), all_ious)

# for i, j in zip(segment_numbers, all_ious):
#     if i <= 10000:
#         plt.text(i**2, j+0.005, '{}'.format(i**2))
#     else:
#         plt.text(i**2, j+0.002, '{}'.format(i**2))

ax[0].set_title(f'Maximum F1-score accuracy for a Given \n  Number of Pixels/Superpixels', fontsize=fs)
ax[0].set_xlabel('Number of Pixels/Superpixels (log scale)', fontsize=fs)
ax[0].set_ylabel('F1-score', fontsize=fs)
ax[0].set_xscale('log')
# ax[0].set_xticks(fontsize=fs, rotation=45)
# ax[0].set_yticks(fontsize=fs)
ax[0].tick_params(axis="x", labelsize=fs, rotation=45) 
ax[0].tick_params(axis="y", labelsize=fs) 
ax[0].legend(loc="lower right", fontsize=fs, title_fontsize=fs)
# for seg in segment_numbers:
#     ax[0].vlines(x=np.power(seg, 2), ymin=0.85, ymax=1, ls=":")
# ax[0].set_tight_layout()

ax[1].set_title(f'MAE for a Given \n  Number of Pixels/Superpixels', fontsize=fs)
ax[1].set_xlabel('Number of Pixels/Superpixels (log scale)', fontsize=fs)
ax[1].set_ylabel('MAE', fontsize=fs)
ax[1].set_xscale('log')
ax[1].tick_params(axis="x", labelsize=fs, rotation=45) 
ax[1].tick_params(axis="y", labelsize=fs) 
ax[1].legend(loc="lower right", fontsize=fs, title_fontsize=fs)
# for seg in segment_numbers:
#     ax[1].vlines(x=np.power(seg, 2), ymin=0.85, ymax=1, ls=":")
# ax[1].set_tight_layout()
# plt.savefig(f'compactness.jpg', bbox_inches='tight')
# ax[1].show()
fig.tight_layout()
plt.show()
    