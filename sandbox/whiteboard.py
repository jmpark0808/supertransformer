from Blocks.swin_encoder_rope import SwinTransformer
from Blocks.swinunet_mix_rope import SwinUTransformer
import torch


res = 56
coeff = 10 
input_dim = 44
window_size = 7
dims = [32, 64, 128, 512]
depths = [2, 2, 6, 2]
heads = [2, 4, 8, 32]
mlp_ratio = 2
classes = 1000
dropout_edge = 0 
dp = 0.1
dropout = 0
load = '/home/eddie/sp_imgnet_ogswin_ape_rope_56370392.ckpt'

supert_imgnet = SwinTransformer(img_size=res, coeff=coeff, in_chans=input_dim, patch_size=1, window_size=window_size,
                                       embed_dim=dims, depths=depths,
                                         num_heads=heads, mlp_ratio=mlp_ratio, num_classes=classes, attn_drop_rate=dropout_edge, 
                                         qkv_bias=False, drop_path_rate=dp, drop_rate=dropout)
dims = [32, 64, 128]
depths = [2, 2, 6]
heads = [2, 4, 8]

supert_swinum = SwinUTransformer(img_size=res, in_chans=input_dim, patch_size=1, window_size=window_size,
                                       embed_dim=dims, depths=depths,
                                         num_heads=heads, mlp_ratio=mlp_ratio, attn_drop_rate=dropout_edge, drop_rate=dropout,
                                         drop_path_rate=dp) 

ckpt = torch.load(load)
for key in list(ckpt['state_dict'].keys()):
    ckpt['state_dict'][key.replace('supert.', '')] = ckpt['state_dict'].pop(key)
supert_imgnet.load_state_dict(ckpt['state_dict'])



checkpoint = torch.load(load)
for key in list(checkpoint['state_dict'].keys()):
    checkpoint['state_dict'][key.replace('supert.', '')] = checkpoint['state_dict'].pop(key)

supert_swinum.load_state_dict(checkpoint['state_dict'], strict=False)


random_input = torch.randn(1, 46, 56, 56)

imgnet_output = supert_imgnet(random_input)
swinum_output = supert_swinum(random_input)

print(torch.sum(torch.abs(imgnet_output-swinum_output)))
