from vit.vit import VisionTransformer
from sklearn.metrics import confusion_matrix, classification_report
from utils_cells import test_report
import torch
from dataset import ImageDataset
from torch.utils.data import DataLoader
from torchvision.models import resnet18
import torch.nn as nn
model = VisionTransformer(image_size=32, in_channels=4, num_classes=4, hidden_dims=[128, 128])

# model = resnet18()
# model.conv1 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

# num_classes = 4
# model.fc = nn.Sequential(
#     nn.Dropout(0.6),  
#     nn.Linear(model.fc.in_features, num_classes)
# )

model_dict = torch.load('models/model_vit8.pth')
model.load_state_dict(model_dict)
dataset = ImageDataset(data_path='test_data')
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

test_report(model, dataloader)

