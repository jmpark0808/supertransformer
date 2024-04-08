import torch
from mamba_ssm import Mamba
from Models.SP_MAMBA import MambaLMHeadModel
from dataclasses import dataclass, field

@dataclass
class MambaConfig:
    d_model: int = 32
    n_layer: int = 6
    vocab_size: int = 50277
    ssm_cfg: dict = None
    rms_norm: bool = True
    residual_in_fp32: bool = True
    fused_add_norm: bool = True
    pad_vocab_size_multiple: int = 8
    tie_embeddings: bool = True


cfg = MambaConfig()

batch, length, dim = 2, 625, 36
x = torch.randn(batch, length, dim).to("cuda")
model = MambaLMHeadModel(
    34, 
    config=MambaConfig

).to("cuda")
print('Model Initialized')
y = model(x)
print('Inferencing')
print(y.shape)