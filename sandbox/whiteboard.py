import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler


with torch.amp.autocast('cuda'):

    batch, sentence_length, embedding_dim = 20, 5, 10
    embedding = torch.ones(batch, sentence_length, embedding_dim, )
    layer_norm = nn.LayerNorm(embedding_dim)
    result = layer_norm(embedding)
    print(result.dtype)
