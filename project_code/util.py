# author : Fahim Tajwar
# necessary util files, like showing plots and histograms
# and showing a tensor as an image

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mp
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image
import matplotlib.image as mpimg
from torch.utils import data

def show_plot(array, x_title = None, y_title = None):
    x_axis = range(len(array))
    y_axis = array

    plt.plot(x_axis, y_axis)
    if x_title != None:
        plt.xlabel(x_title)
    if y_title != None:
        plt.ylabel(y_title)

    plt.show()

def plot_bar_graph_from_map(map, x_label, y_label, label_for_each_class):
    plt.bar(range(len(map)), list(map.values()), align='center')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xticks(range(len(map)), label_for_each_class)

def show_tensor_as_image(tensor, title = None):
    image = tensor.clone().cpu()
    image = image.view(*tensor.size())
    image = transforms.ToPILImage()(image)
    plt.imshow(image)

    if title is not None:
        plt.title(title)
    plt.pause(5)

def plot_x_vs_y(x_axis, y_axis, x_title = None, y_title = None):
    plt.plot(x_axis, y_axis)
    if x_title != None:
        plt.xlabel(x_title)
    if y_title != None:
        plt.ylabel(y_title)

    plt.show()

def flatten(x):
    N = x.shape[0]
    return x.view(N,-1)