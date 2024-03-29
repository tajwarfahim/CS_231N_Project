# author : Fahim Tajwar
# necessary util files, like showing plots and histograms
# and showing a tensor as an image, splitting dataset, setting initial weights
# that are generally used for various purposes, not restricted to any single class

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
import math as math

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
    plt.show()

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

def get_dataset_split(Y_label, train_fraction = 0.8, validation_fraction = 0.1, test_fraction = 0.1):
    ids = np.array(list(Y_label.keys()))
    random_indices = np.random.choice(len(ids), len(ids), replace = False)

    train_num = int(math.floor(train_fraction * len(ids)))
    training_indices = random_indices[0: train_num]
    training_labels = ids[training_indices]

<<<<<<< HEAD:util.py
    validation_num = int(math.floor(validation_fraction * len(ids)))
    #validation_indices = random_indices[train_num : train_num + validation_num]
    validation_indices = random_indices[0 : validation_num]
    validation_labels = ids[validation_indices]

    test_num = int(math.floor(test_fraction * len(ids)))
    #testing_indices = random_indices[train_num + validation_num : train_num + validation_num + test_num]
    testing_indices = random_indices[train_num - test_num : train_num]
=======
    validation_num = math.floor(validation_fraction * len(ids))
    validation_indices = random_indices[train_num : train_num + validation_num]
    validation_labels = ids[validation_indices]

    test_num = math.floor(test_fraction * len(ids))
    testing_indices = random_indices[train_num + validation_num : train_num + validation_num + test_num]
>>>>>>> 5d16dde1393905958c62c8b419e68f6714d7ab3a:project_code/util.py
    testing_labels = ids[testing_indices]

    return training_labels, validation_labels, testing_labels


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        y = m.in_features
        m.weight.data.normal_(0.0,1/np.sqrt(y))
        m.bias.data.fill_(0)
        

# only works for one on one maps
def reverse_map(dictionary):
    new_dictionary = {}
    for i in dictionary:
        new_dictionary[dictionary[i]] = i
    
    return new_dictionary
