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
from util import *
from vgg import *

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

# our first design of 2D convnet
# architecture: [conv -> relu -> pool] x N -> affine x M -> weighted_cross_entropy_loss
class Our2DConvNetDesign1(nn.Module):
    def __init__(self, num_classes = 3, input_channels = 3, input_size = 224):
        super(Our2DConvNetDesign1, self).__init__()
        out_channel_1 = 16
        out_channel_2 = 32
        out_channel_3 = 64

        conv_layer_1 = nn.Sequential(
            nn.Conv2d(input_channels, out_channel_1, kernel_size = 5, padding = 2),
            nn.BatchNorm2d(out_channel_1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        conv_layer_2 = nn.Sequential(
            nn.Conv2d(out_channel_1, out_channel_2, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(out_channel_2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        conv_layer_3 = nn.Sequential(
            nn.Conv2d(out_channel_2, out_channel_3, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(out_channel_3),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        linear_layer_size = ((int(input_size / (2 * 2 * 2))) ** 2) * out_channel_3
        fc = nn.Linear(linear_layer_size, num_classes)

        self.model = nn.Sequential(
            conv_layer_1,
            conv_layer_2,
            conv_layer_3,
            Flatten(),
            fc
        )

    def forward(self, X):
        return self.model(X)


# our second model
# architecture : [conv -> relu -> conv -> relu -> max pool] x M -> affine x N -> cross entropy loss
class Our2DConvNetDesign2(nn.Module):
    def __init__(self, num_classes = 3, input_channels = 3, input_size = 224):
        super(Our2DConvNetDesign2, self).__init__()
        out_channel_1 = {1 : 16, 2 : 16}
        out_channel_2 = {1 : 32, 2 : 32}
        out_channel_3 = {1 : 64, 2 : 64}

        conv_layer_1 = nn.Sequential(
            nn.Conv2d(input_channels, out_channel_1[1], kernel_size = 5, padding = 2),
            nn.BatchNorm2d(out_channel_1[1]),
            nn.ReLU(),
            nn.Conv2d(out_channel_1[1], out_channel_1[2], kernel_size  = 3, padding = 1),
            nn.BatchNorm2d(out_channel_1[2]),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        conv_layer_2 = nn.Sequential(
            nn.Conv2d(out_channel_1[2], out_channel_2[1], kernel_size = 5, padding = 2),
            nn.BatchNorm2d(out_channel_2[1]),
            nn.ReLU(),
            nn.Conv2d(out_channel_2[1], out_channel_2[2], kernel_size  = 3, padding = 1),
            nn.BatchNorm2d(out_channel_2[2]),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        conv_layer_3 = nn.Sequential(
            nn.Conv2d(out_channel_2[2], out_channel_3[1], kernel_size = 5, padding = 2),
            nn.BatchNorm2d(out_channel_3[1]),
            nn.ReLU(),
            nn.Conv2d(out_channel_3[1], out_channel_3[2], kernel_size  = 3, padding = 1),
            nn.BatchNorm2d(out_channel_3[2]),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        linear_layer_size = ((int(input_size / (2 * 2 * 2))) ** 2) * out_channel_3[2]
        fc = nn.Linear(linear_layer_size, num_classes)

        self.model = nn.Sequential(
            conv_layer_1,
            conv_layer_2,
            conv_layer_3,
            Flatten(),
            fc
        )

    def forward(self, X):
        return self.model(X)

# deep neural network architectures
class VGG16(nn.Module):
    def __init__(self, num_classes):
        super(VGG16, self).__init__()
        self.model = vgg16_bn()

    def forward(self, x):
        return self.model(x)


# 3D convolutional networks

# first, a very simple one, just as a baseline
class Simple3DConvNet(nn.Module):
    def __init__(self, num_classes = 3, input_channels = 3, input_size = 224, input_time_depth = 5):
        super(Simple3DConvNet, self).__init__()
        output_channel = 8
        assert(input_size % 2 == 0)

        layer1 = nn.Sequential(
            nn.Conv3d(input_channels, output_channel, kernel_size = 5, stride = 1, padding = 2),
            nn.BatchNorm3d(output_channel),
            nn.ReLU(),
        )

        linear_layer_size = input_size * input_size * input_time_depth * output_channel
        fc = nn.Linear(linear_layer_size, num_classes)

        self.model = nn.Sequential(
            layer1,
            Flatten(),
            fc
        )

    def forward(self, x):
        return self.model(x)
