from Models.SP_GAT import SP_GATv2
from Models.SP_TFM import SP_TFM_REL
from Blocks.GraphBlocks import GATv2
from Blocks.TransformerBlocks import PosAttention
import torch
import time
import numpy as np

gat = SP_GATv2(26, 16, 0, 8, 6).cuda()
tfm = SP_TFM_REL(26, 1, 16, 8, 6, 0).cuda()



dummy_input = torch.randn(8, 625, 28).cuda()
dummy_adj = torch.randn(16, 625, 625).cuda()

gat_run_times = []
tfm_run_times = []





for _ in range(1000):
    start = time.time()
    tfm(dummy_input, None, None)
    end = time.time()
    tfm_run_times.append(end-start)


for _ in range(1000):
    start = time.time()
    gat(dummy_input)
    end = time.time()
    gat_run_times.append(end-start)

print(f'GAT {np.mean(gat_run_times)}, TFM {np.mean(tfm_run_times)}')


gat = GATv2(28, 16, 0, 8, 0.2, False).cuda()
tfm = PosAttention(28, 1, 8, 16, 0).cuda()

gat_run_times = []
tfm_run_times = []





for _ in range(1000):
    start = time.time()
    tfm(dummy_input, None, None, None)
    end = time.time()
    tfm_run_times.append(end-start)


for _ in range(1000):
    start = time.time()
    gat(dummy_input, None)
    end = time.time()
    gat_run_times.append(end-start)

print(f'GAT Block {np.mean(gat_run_times)}, TFM Block {np.mean(tfm_run_times)}')