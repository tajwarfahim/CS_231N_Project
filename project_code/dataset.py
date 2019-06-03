# author: Fahim Tajwar

# ideas taken from : https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
# and https://pytorch.org/docs/stable/torchvision/transforms.html
# and https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

# however, none of the codes exactly match, so the code here was implemented line by line by the Author
# taking inspiration from the sources

# import and simple transformation initializations
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from project_code.label_loader import *
import numpy as np
from torch.utils import data

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

scaler = transforms.Resize((224, 224))
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
to_tensor = transforms.ToTensor()

# vannila dataset
class Dataset(data.Dataset):
  def __init__(self, list_IDs, labels):
        self.labels = labels
        self.list_IDs = list_IDs

  def __len__(self):
        return len(self.list_IDs)

  def __getitem__(self, index):
        ID = self.list_IDs[index]
        y = self.labels[ID]
        X = X_data[ID]
        return X, y

# dataset with augmentation possible
class Augmented_Dataset(data.Dataset):
    def __init__(self, list_IDs, labels, transform = None):
        self.labels = labels
        self.list_IDs = list_IDs
        self.transform = transform

    def __len__(self):
          'Denotes the total number of samples'
          return len(self.list_IDs)

    def __getitem__(self, index):
        ID = self.list_IDs[index]
        y = self.labels[ID]
        X = X_data[ID]

        if self.transform != None:
            X = self.transform(X)

        return X, y

# experiments with different type of augmentation
transform_horizontal_flip = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

transfor_vertical_flip = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

transform_flip = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

transform_augmentation = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20)
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])