from dataset import *
from utils_cells import *
from torchvision.transforms import transforms
from model import CNN_model
import torch

model = CNN_model(4, 4)

x = torch.randn(4, 32, 32)
x = x.to('cuda')  # Move input tensor to the GPU
model = model.to('cuda')  # Move the entire model to the GPU
out = model(x)
