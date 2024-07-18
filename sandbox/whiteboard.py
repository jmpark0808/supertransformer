import torch
from fvcore.nn import FlopCountAnalysis, flop_count_table, parameter_count
from Blocks.TransformerBlocks import Transformer, Attention
from Blocks.performer import Token_performer
from performer_pytorch import Performer
from performer_pytorch import SelfAttention

from zeta.nn import MambaBlock

block = MambaBlock(dim=128, depth=1)

# print(output.shape)

inp = torch.randn([1, 1024, 128])
supert = Token_performer(128, 128, 1)
supert_1 = Transformer(128, 1, 1, 128, 128, 0, 0)


# model = Performer(
#     dim = 128,
#     depth = 1,
#     heads = 1,
#     dim_head =128,
#     causal = False
# )
model = SelfAttention(dim=128, causal=False, heads=4, dim_head=32)
# supert_2 = MixerModel(128, 1, )
# supert_1 = Attention(128, 1, 128, 0, 0)

flops = FlopCountAnalysis(supert_1, inp)
print('Transformer', flops.total())

flops = FlopCountAnalysis(block, inp)
print('Mamba', flops.total())

flops = FlopCountAnalysis(model, inp)
print('Performer packaged', flops.total())

flops = FlopCountAnalysis(supert, inp)
print('Performer impl', flops.total())





