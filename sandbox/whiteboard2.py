import os
from util.util import S_object, eval_e, S_region
import imageio.v2 as imageio
import numpy as np
import torch
data_directory = '/home/eddie/Datasets'
pred_directory = '/home/eddie/Qualitative/iNas/'
datasets = ['DUTS-TE', 'DUTS-OMRON', 'ECSSD', 'HKU-IS', 'PASCAL']

for dataset in datasets:
    preds = sorted([os.path.join(pred_directory, dataset, x) for x in os.listdir(os.path.join(pred_directory, dataset))])
    if dataset == 'DUTS-TE':
        labels = sorted([os.path.join(data_directory, 'DUTS', dataset,'Mask',  x) for x in os.listdir(os.path.join(data_directory, 'DUTS', dataset, 'Mask'))])
    else:

        labels = sorted([os.path.join(data_directory, dataset,'Mask', x) for x in os.listdir(os.path.join(data_directory, dataset, 'Mask'))])

    e_measure_scores = torch.zeros(255).cuda()
    s_measure_q = 0.0
    mean_num = 0
    for pred, label in zip(preds, labels):
        img_pred = imageio.imread(pred, mode='L')/255.
        img_label = imageio.imread(label, mode='L')/255.

        res = torch.tensor(img_pred).cuda()
        gt = torch.tensor(img_label).cuda()
        e_measure_scores += eval_e(res, gt, 255)
        y = gt.mean()
        if y == 0:
            x = res.mean()
            Q = 1.0 -x
        elif y == 1:
            x = res.mean()
            Q = x
        else:
            gt[gt>=0.5] = 1
            gt[gt<0.5] = 0
            Q = 0.5 * S_object(res, gt) + (1-0.5) * S_region(res, gt)
            if Q.item() < 0:
                Q = torch.FloatTensor([0.0])
        s_measure_q += Q.item()
        mean_num += 1
        
    print(e_measure_scores.max()/mean_num)
    print(s_measure_q/mean_num)


