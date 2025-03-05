import numpy as np

cvpr_dataset = '/home/eddie/Datasets/DUTS-3136/DUTS/DUTS-TR/SPFFFT/ILSVRC2012_test_00000004_features.npy'
new_dataset = '/home/eddie/Datasets/DUTS/DUTS-TR/SPFFFT/ILSVRC2012_test_00000004_features.npy'

cvpr = np.load(cvpr_dataset)
new = np.load(new_dataset)


for i in range(cvpr.shape[0]):
    
    print(new[i, :].shape)
    cvpr_amp = cvpr[i, 8:18]
    cvpr_phase = cvpr[i, 18:28]
    cvpr_lbp = cvpr[i, 28:]
    new_amp = new[i, 8:23]
    new_phase = new[i, 23:38]
    new_moments = new[i, 38:46]
    new_lbp = new[i, 46:]

    print(cvpr_amp)
    print(cvpr_phase)
    print(cvpr_lbp)

    print(new_amp)
    print(new_phase)
    print(new_moments)
    print(new_lbp)
    assert(0)