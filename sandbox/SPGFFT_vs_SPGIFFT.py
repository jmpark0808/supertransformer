import numpy as np
import os
import sys
sys.path.insert(0, '/mnt/pegasus/waterloo/supertransformer')
from dataset.superpixel_pyg_image import SPDataset as SPGIDataset
from dataset.superpixel_pyg import SPDataset as SPGDataset
from torch_geometric.loader import DataLoader as GDL
from torch.utils.data import DataLoader as DL
import torch
import torch.nn as nn
from torch_geometric.nn.dense.linear import Linear
from Models.SP_TFM_PyG import SP_TFM_PyG
from Models.SP_TFM import SP_TFM_REL
from Models.SP_GAT_PyG import SP_GAT_PyG
import time



train_dir = '/mnt/dragon/Datasets/DUTS/DUTS-TR/'


image_list = np.array(sorted([os.path.join('{}/Image'.format(train_dir), f) for f in os.listdir('{}/Image'.format(train_dir))]))
mask_list = np.array(sorted([os.path.join('{}/Mask'.format(train_dir), f) for f in os.listdir('{}/Mask'.format(train_dir))]))

image_list = image_list[:int(len(image_list)*0.01)]
mask_list = mask_list[:int(len(mask_list)*0.01)]


spf_dataset = SPGIDataset(image_list, mask_list, 625, 256, 10, 'SPGIFFT', True, 10, False, False, 7)
spg_dataset = SPGDataset(image_list, mask_list, 625, 256, 10, 'SPGFFT', True, 10, False, False, None, None, 7)

spf_loader = GDL(spf_dataset, 3, False, num_workers=4)
spg_loader = GDL(spg_dataset, 3, False, num_workers=4)

pyg_gat = SP_GAT_PyG(26, 16, None, 0, 8, 6, 625)
pyg_gat.eval()

for a, b in zip(spf_loader, spg_loader):
    spgi_features = a[0]
    spgi_seq_mask = a[1]
    spgi_segments = a[2]
    spgi_mask = a[3]
   

    spg_features = b[0]

    spg_seq_mask = b[1]
    spg_segments = b[2]
    spg_mask = b[3]

    print(torch.sum(torch.abs(spgi_features.edge_index-spg_features.edge_index)))

    # features = batch[0]
    # node_num = batch[1]
    # segments = batch[2]
    # mask = batch[3]

    # print(spgi_features.x.size(), spg_features.x.size())
    # print(spgi_features.x.dtype, spg_features.x.dtype)
    # print(spgi_features.edge_index.dtype,spg_features.edge_index.dtype)
    # print(spgi_seq_mask.dtype,spg_seq_mask.dtype)
    # print(spgi_segments.dtype,spg_segments.dtype)
    # print(spgi_mask.dtype,spg_mask.dtype)

    with torch.no_grad():
        spgi_output = pyg_gat(spgi_features)
        spg_output = pyg_gat(spg_features)
        diff = torch.sum(torch.abs(spgi_output-spg_output))
        print(diff)


    if torch.sum(torch.abs(spgi_features.x-spg_features.x)) != 0 or \
        torch.sum(torch.abs(spgi_features.edge_index-spg_features.edge_index)) != 0 or \
        torch.sum(torch.abs(spgi_seq_mask-spg_seq_mask)) != 0 or \
        torch.sum(torch.abs(spgi_segments-spg_segments)) != 0 or \
        torch.sum(torch.abs(spgi_mask-spg_mask)) != 0:

        print(f'Feature diff {torch.sum(torch.abs(spgi_features.x-spg_features.x))}')
        print(f'EI diff {torch.sum(torch.abs(spgi_features.edge_index-spg_features.edge_index))}')
        print(f'Seq mask diff {torch.sum(torch.abs(spgi_seq_mask-spg_seq_mask))}')
        print(f'Segments diff {torch.sum(torch.abs(spgi_segments-spg_segments))}')
        print(f'Mask diff {torch.sum(torch.abs(spgi_mask-spg_mask))}')
        assert(0)



# image_list = image_list[:1]
# mask_list = mask_list[:1]


# spf_dataset = SPGIDataset(image_list, mask_list, 625, 256, 10, 'SPGIFFT', True, 10, False, False, 7)

# spf_loader = GDL(spf_dataset, 1, False, num_workers=4)

# inputs = []
# outputs = []
# for _ in range(10):
#     for a in spf_loader:
#         spgi_features = a[0]
#         spgi_seq_mask = a[1]
#         spgi_segments = a[2]
#         spgi_mask = a[3]

#         inputs.append(spgi_features.x.reshape(-1))
#         outputs.append(spgi_seq_mask)

# inputs = torch.stack(inputs, dim=0)
# outputs = torch.stack(outputs, dim=0)
# print(inputs.size())
# inputs = torch.std(inputs, dim=0)
# outputs = torch.std(outputs, dim=0)
# print(torch.sum(inputs))
# print(torch.sum(outputs))
    

assert(0)






def init_weights(m):
    if isinstance(m, nn.Linear):
        m.weight.data.fill_(0.5)
        if m.bias is not None:
            m.bias.data.fill_(0.1)
    if isinstance(m, Linear):
        m.weight.data.fill_(0.5)
        if m.bias is not None:
            m.bias.data.fill_(0.1)
    if isinstance(m, nn.LayerNorm):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)

pyg_tfm = SP_TFM_PyG(26, 16, None, 0, 8, 6, 625).cuda()
pyg_tfm.apply(init_weights)
pyg_tfm.eval()

pyt_tfm = SP_TFM_REL(26, 7, 16, 8, 6, 0).cuda()
pyt_tfm.apply(init_weights)
pyt_tfm.eval()

start = time.time()
pyg_times = []
for ind, b in enumerate(spg_loader):
    end = time.time()
    pyg_times.append(end-start)
    start = time.time()
    
print('PyG', np.mean(pyg_times))
start = time.time()
pyt_times = []
for ind, a in enumerate(spf_loader):
    end = time.time()
    pyt_times.append(end-start)
    start = time.time()
print('PyT', np.mean(pyt_times))
assert(0)

for a, b in zip(spf_loader, spg_loader):
    a_features = a['features'].cuda()
    a_seq_mask = a['seq_mask']
    a_segments = a['segments']
    a_mask = a['mask'].cpu()
    a_adj = a['neighbor_array'].cuda()


    a_edge_index = np.array(np.nonzero(a_adj.detach().cpu().numpy()))
 

    pyg_input = b[0].cuda()
    batch_ind = []
    for idx, node in enumerate(b[1]):
        batch_ind += [idx]*node
    batch_ind = torch.tensor(batch_ind).cuda()

    start = time.time()
    out_pyg = pyg_tfm(pyg_input, batch_ind)
    end = time.time()

    # print(out_pyg[0])
    print(f'PyG Time: {end-start}')

    start = time.time()
    out_pyt = pyt_tfm(a_features, a_adj, None)
    end = time.time()
    print(f'PyT Time: {end-start}')
    # print(out_pyt[0])
    # print(torch.sum(torch.abs(out_pyg-out_pyt)))
    # print(len(a_edge_index[0, :]), len(b_edge_index[0, :]))
    # assert(0)
    


