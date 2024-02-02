
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import matplotlib as mpl


spg_1 = pd.read_csv('/mnt/hdd/Experiments/spg_vs_spgi/spg/run-spg_1207-000516_42990997-tag-Test Max F Score.csv')
spg_2 = pd.read_csv('/mnt/hdd/Experiments/spg_vs_spgi/spg/run-spg_1207-002350_42991057-tag-Test Max F Score.csv')
spg_3 = pd.read_csv('/mnt/hdd/Experiments/spg_vs_spgi/spg/run-spg_1207-020944_42991168-tag-Test Max F Score.csv')
spg_4 = pd.read_csv('/mnt/hdd/Experiments/spg_vs_spgi/spg/run-spg_1207-022530_42991233-tag-Test Max F Score.csv')
spg_5 = pd.read_csv('/mnt/hdd/Experiments/spg_vs_spgi/spg/run-spg_1207-063359_42991286-tag-Test Max F Score.csv')

spgi_1 = pd.read_csv('/mnt/hdd/Experiments/spg_vs_spgi/spgi/run-spgi_1207-002405_42991422-tag-Test Max F Score.csv')
spgi_2 = pd.read_csv('/mnt/hdd/Experiments/spg_vs_spgi/spgi/run-spgi_1207-141403_42991466-tag-Test Max F Score.csv')
spgi_3 = pd.read_csv('/mnt/hdd/Experiments/spg_vs_spgi/spgi/run-spgi_1208-075743_42991499-tag-Test Max F Score.csv')
spgi_4 = pd.read_csv('/mnt/hdd/Experiments/spg_vs_spgi/spgi/run-spgi_1208-190227_42991542-tag-Test Max F Score.csv')
spgi_5 = pd.read_csv('/mnt/hdd/Experiments/spg_vs_spgi/spgi/run-spgi_1209-014707_42991644-tag-Test Max F Score.csv')


n_lines = 10
c = np.arange(1, n_lines + 1)
norm = mpl.colors.Normalize(vmin=c.min(), vmax=c.max())
cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.Blues)
cmap.set_array([])

plt.plot(spg_1['Step'], spg_1['Value'], c=cmap.to_rgba(6))
plt.plot(spg_2['Step'], spg_2['Value'], c=cmap.to_rgba(7))
plt.plot(spg_3['Step'], spg_3['Value'], c=cmap.to_rgba(8))
plt.plot(spg_4['Step'], spg_4['Value'], c=cmap.to_rgba(9))
plt.plot(spg_5['Step'], spg_5['Value'], c=cmap.to_rgba(10), label='Graph')

norm = mpl.colors.Normalize(vmin=c.min(), vmax=c.max())
cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.Reds)
cmap.set_array([])

plt.plot(spgi_1['Step'], spgi_1['Value'], c=cmap.to_rgba(6))
plt.plot(spgi_2['Step'], spgi_2['Value'], c=cmap.to_rgba(7))
plt.plot(spgi_3['Step'], spgi_3['Value'], c=cmap.to_rgba(8))
plt.plot(spgi_4['Step'], spgi_4['Value'], c=cmap.to_rgba(9))
plt.plot(spgi_5['Step'], spgi_5['Value'], c=cmap.to_rgba(10), label='Image')

plt.ylabel('F score')
plt.xlabel('Training Iteration')
plt.title('Comparing 5 runs of Graph vs Image')
plt.legend()
plt.show()
