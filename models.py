## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):
    fc_inputs = 7 * 7 * 256

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        # Transformation input formula : (InputSize - FilterSize + 2 Padding) / Stride + 1
        # 224x224x1
        self.conv1 = nn.Conv2d(1, 16, 4, padding=1)
        # (224 - 2 + 2 * 0) / 2 + 1 = 112x112x16
        self.conv2 = nn.Conv2d(16, 32, 4, padding=1)
        # (112 - 2 + 2 * 0) / 2 + 1 = 56x56x32
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        # (56 - 2 + 2 * 0) / 2 + 1 = 28x28x64
        self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
        # (28 - 2 + 2 * 0) / 2 + 1 = 14x14x128
        self.conv5 = nn.Conv2d(128, 256, 2, padding=1)
        # (14 - 2 + 2 * 0) / 2 + 1 = 7x7x256

        # max pooling
        self.pool = nn.MaxPool2d(2, 2, padding=0)

        # dropout
        self.conv_dropout = nn.Dropout2d(p=0.5)

        self.fc_dropout = nn.Dropout(p=0.5)

        # full connected
        self.fc1 = nn.Linear(Net.fc_inputs, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 136)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        
        # Conv pass
        x = self.pool(F.relu(self.conv1(x)))
        x = self.conv_dropout(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.conv_dropout(x)
        x = self.pool(F.relu(self.conv3(x)))
        x = self.conv_dropout(x)
        x = self.pool(F.relu(self.conv4(x)))
        x = self.conv_dropout(x)
        x = self.pool(F.relu(self.conv5(x)))
        
        # flatten
        x = x.view(x.shape[0], -1)

        # FC pass
        x = F.relu(self.fc1(x))
        x = self.fc_dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc_dropout(x)
        x = self.fc3(x)
        # a modified x, having gone through all the layers of your model, should be returned
        return x
