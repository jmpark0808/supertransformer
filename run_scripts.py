from legacy.dsc import DSC
import torch


test = torch.rand(1, 3, 256, 256)
model = DSC()

model(test)