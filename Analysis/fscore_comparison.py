
import os
from glob import glob

egnet_dir = '/home/abcd/abcde/supertransformer/results/egnet_results'
sp_tfm_dir = '/home/abcd/abcde/supertransformer/results/sp_tfm_results'
d = {}
for i in range(5019):
    egnet_files = glob(os.path.join(egnet_dir, f'{i}_*'))
    sp_tfm_files = glob(os.path.join(sp_tfm_dir, f'{i}_*'))

    assert len(egnet_files) == 1, f'{egnet_files}'

    assert len(sp_tfm_files) == 1, f'{sp_tfm_files}'

    egnet_file = egnet_files[0]
    sp_tfm_file = sp_tfm_files[0]


    fscore_eg = float(egnet_file.split('_')[-1].split('.png')[0])
    fscore_sp = float(sp_tfm_file.split('_')[-1].split('.png')[0])


    d[egnet_file] = fscore_eg-fscore_sp

sorted_dict = [k for k, v in sorted(d.items(), key=lambda item: item[1])]
print(sorted_dict[:5])
print(sorted_dict[-5:])
