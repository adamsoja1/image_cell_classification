import torch

model = torch.load('vit.bin')

print(model.values())