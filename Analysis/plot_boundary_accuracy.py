import pickle
import matplotlib.pyplot as plt

with open('segments_plot_data.pkl', 'rb') as f:
    loaded_dict = pickle.load(f)
plt.figure(figsize=(10,10))
segment_numbers = loaded_dict['segment_numbers']
compactness = [0.1, 1, 10, 50]
for compact in compactness:
    all_ious = loaded_dict[compact]
    plt.plot(segment_numbers, all_ious, label=f'{compact}')
    plt.scatter(segment_numbers, all_ious)
    if compact == 10:
        for i, j in zip(segment_numbers, all_ious):
            if i <= 10000:
                plt.text(i, j+0.005, '{}'.format(i))
            else:
                plt.text(i, j+0.002, '{}'.format(i))

fs = 20
plt.title(f'Upperbound F1-score accuracy as a function of segmentation number', fontsize=fs)
plt.xlabel('Number of Segments (log scale)', fontsize=fs)
plt.ylabel('F1-score', fontsize=fs)
plt.xscale('log')
plt.xticks(fontsize=fs, rotation=45)
plt.yticks(fontsize=fs)
plt.legend(loc="lower right", fontsize=fs, title='Compactness', title_fontsize=fs)
plt.vlines(x=segment_numbers, ymin=0.93, ymax=1, ls=":")
plt.savefig(f'compactness.jpg')
    