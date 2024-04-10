# import timm

# model = timm.create_model('vit_base_patch32_224_in21k', pretrained=True, num_classes=1000)
# print(model)

from transformers import ViTImageProcessor, ViTModel
from PIL import Image
import requests
import torch

url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)

processor = ViTImageProcessor.from_pretrained('google/vit-base-patch32-224-in21k')
model = ViTModel.from_pretrained('google/vit-base-patch32-224-in21k')

inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)
last_hidden_state = outputs.last_hidden_state

import torch.nn as nn

cos = nn.CosineSimilarity(dim=0)
s = model.embeddings.position_embeddings.shape
pos_patch = model.embeddings.position_embeddings.view(*s[1:])[1:].view(7, 7,-1)
import matplotlib.pyplot as plt
import numpy as np
fig, ax = plt.subplots(7, 7)
for k in range(7):
    for l in range(7):
        patches = []
        for i in range(7):
            for j in range(7):
                patches.append(torch.sqrt(torch.sum(torch.pow(pos_patch[k, l]-pos_patch[i, j], 2))).detach().cpu().numpy())
        ax[k, l].imshow(np.array(patches).reshape(7, 7), cmap='hot')
        ax[k, l].set_xticks([])
        ax[k, l].set_yticks([])
        if l == 0:
            ax[k, l].set_ylabel(f'{k+1}')
        if k == 6:
            ax[k, l].set_xlabel(f'{l+1}')
        


fig.suptitle('Vision Transformer Positional Encoding Euclidean Distance')
plt.show()

