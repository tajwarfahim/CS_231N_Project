# author: Fahim Tajwar
# contains different models for testing

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from torch.utils import data
from project_code.util import *

# baseline model
class LogisticRegression(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, x):
        flat = flatten(x)
        out = self.linear(flat)
        return out

# simple convnet design, second baseline
class SimpleConvNet(nn.Module):
    def __init__(self, num_classes = 3, input_channels = 3, input_size = 224):
        super(SimpleConvNet, self).__init__()
        output_channel = 16
        assert(input_size % 2 == 0)
        self.layer1 = nn.Sequential(
            nn.Conv2d(input_channels, output_channel, kernel_size = 5, stride = 1, padding = 2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        half_size = int(input_size / 2)
        linear_layer_size = half_size * half_size * output_channel
        self.fc = nn.Linear(linear_layer_size, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

# nn module to be used in other Sequential models developped later
class Flatten(nn.Module):
    def forward(self, X):
        return flatten(X)

# our design of 2D convnet
# architecture: [conv -> relu -> pool] x N -> affine x M ->
#class Our2DConvNetDesign1(nn.Module):
    #def
