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
  def __init__(self, X_data, list_IDs, labels, well_id_to_image_id_map):
        self.labels = labels
        self.list_IDs = list_IDs
        self.X_data = X_data
        self.well_id_to_image_id_map = well_id_to_image_id_map

  def __len__(self):
        return len(self.list_IDs)

  def __getitem__(self, index):
        ID = self.list_IDs[index]
        y = self.labels[ID]
        image_id = self.well_id_to_image_id_map[ID]
        X = self.X_data[image_id]
        return X, y

# dataset with augmentation possible
class Augmented_Dataset(data.Dataset):
    def __init__(self, X_data, list_IDs, labels, well_id_to_image_id_map, transform = None):
        self.labels = labels
        self.list_IDs = list_IDs
        self.transform = transform
        self.X_data = X_data
        self.well_id_to_image_id_map = well_id_to_image_id_map

    def __len__(self):
          return len(self.list_IDs)

    def __getitem__(self, index):
        ID = self.list_IDs[index]
        y = self.labels[ID]
        image_id = self.well_id_to_image_id_map[ID]
        X = self.X_data[image_id]

        if self.transform != None:
            X = self.transform(X)

        return X, y

# experiments with different type of augmentation
transform_horizontal_flip = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

transform_horizontal_flip_normalization = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

transfor_vertical_flip = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
    ])

transfor_vertical_flip_normalization = transforms.Compose([
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
    ])

transform_flip_normalization = transforms.Compose([
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
        transforms.RandomRotation(20),
        transforms.ToTensor(),
    ])

transform_augmentation_normalization = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

transform_normalization = transforms.Compose([
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# our dedicated final choice
# this one performs best
final_transformation_choice = transform_flip_normalization
