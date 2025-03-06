


from Blocks.swinunet_mix_rope import SwinUTransformer as ROPE
from Blocks.swinunet_mix_ape import SwinUTransformer

rpe = ROPE(img_size=72, in_chans=34, patch_size=1, window_size=9,
                                       embed_dim=[48, 96, 176], depths=[2, 2, 6],
                                         num_heads=[3, 6, 11], mlp_ratio=2, attn_drop_rate=0, drop_rate=0,
                                         qkv_bias=False, drop_path_rate=0.1).cuda()


ape = SwinUTransformer(img_size=72, in_chans=34, patch_size=1, window_size=9,
                                       embed_dim=[32, 64, 128], depths=[2, 2, 6],
                                         num_heads=[2, 4, 8], mlp_ratio=2, attn_drop_rate=0, drop_rate=0,
                                         qkv_bias=False, drop_path_rate=0.1).cuda()


import torch
inp = torch.randn(1, 36, 72, 72).cuda()

rpe.eval()
ape.eval()
import time
rope_times = []
ape_times=  []
with torch.no_grad():
    for _ in range(1000):
        start = time.time()
        rpe(inp)
        end = time.time()
        rope_times.append(end-start)
        start = time.time()
        ape(inp)
        end = time.time()
        ape_times.append(end-start)

import numpy as np

print(np.mean(rope_times))
print(np.mean(ape_times))
