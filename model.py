import torch
import torch.nn as nn

class CNN_model(nn.Module):
    def __init__(self, in_channels, classes):
        super(CNN_model, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv2 = nn.Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv3 = nn.Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv4 = nn.Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv5 = nn.Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv6 = nn.Conv2d(32, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.fc1 = nn.Linear(16 * 32 * 32, 64)  # Fix this line based on the actual output size
        self.fc2 = nn.Linear(64, classes)

        self.relu = nn.ReLU()
        self.softmax = nn.LogSoftmax(dim=1)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        x = self.relu(self.conv6(x))

        # Calculate the actual size after convolutional layers and update fc1 input size
        x = self.flatten(x)
        fc1_input_size = x.size(1)
        
        self.fc1 = nn.Linear(fc1_input_size, 64)
        self.fc1 = self.fc1.to('cuda')
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))

        return self.softmax(x)




