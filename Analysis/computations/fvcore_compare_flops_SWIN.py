import sys
sys.path.insert(0, '/home/eddie/waterloo/supertransformer')

from Models.SP_TFM import SP_TFM_REL
from Models.SP_SWIN import SP_SWIN, SP_SWINU
from fvcore.nn import FlopCountAnalysis, flop_count_table
import torch
# Regular TFM

model = SP_TFM_REL(36, 1, 16, 8, 6, 0, 0).cuda()
inp = torch.randn([1, 1024, 38]).cuda()
flops = FlopCountAnalysis(model, inp)
print(flop_count_table(flops))

model = SP_SWIN(36, 16, 8, 2, 0, 0).cuda()
flops = FlopCountAnalysis(model, inp)
print(flop_count_table(flops))

model = SP_SWINU(36, 16, 8, 2, 0, 0).cuda()
flops = FlopCountAnalysis(model, inp)
print(flop_count_table(flops))
