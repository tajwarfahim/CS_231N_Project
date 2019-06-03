# author : Fahim Tajwar and Sandhini Agarwal

# imports and initializations
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

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

scaler = transforms.Resize((224, 224))
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
to_tensor = transforms.ToTensor()


# helper functions
def get_well_specific_string_prefix(well_num):
    well_str = ""
    if well_num <= 9:
        well_str = "000"
    elif well_num <= 99:
        well_str = "00"
    elif well_num <= 999:
        well_str = "0"

    return well_str

def get_day_specific_string_prefix(day):
    day_str = "_day"
    if day <= 9:
        day_str = "_day0"

    return day_str

def list_of_image_names(min_well, max_well, min_day, max_day, prefix, suffix):
    well_values = range(min_well, max_well + 1)
    day_values = range(min_day, max_day + 1)

    image_names = []
    for well_num in well_values:
        for day in day_values:
            well_str = get_well_specific_string_prefix(well_num)
            day_str = get_day_specific_string_prefix(day)
            name = prefix + well_str + str(well_num) + day_str + str(day) + suffix
            image_names.append(name)

    return image_names

def list_of_image_names_from_set_of_wells(all_well_ids, min_day, max_day, prefix, suffix):
    day_values = range(min_day, max_day + 1)
    image_names = []
    for well_num in all_well_ids:
        for day in day_values:
            well_str = get_well_specific_string_prefix(well_num)
            day_str = get_day_specific_string_prefix(day)
            name = prefix + well_str + str(well_num) + day_str + str(day) + suffix
            image_names.append(name)

    return image_names

# source: https://gist.github.com/soply/f3eec2e79c165e39c9d540e916142ae1
def show_images(images, cols = 1, titles = None):
    assert((titles is None) or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None: titles = ['Image (%d)' % i for i in range(1,n_images + 1)]
    fig = plt.figure()
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, np.ceil(n_images/float(cols)), n + 1)
        plt.imshow(image)
        a.set_title(title)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.show()

def read_images(min_well, max_well, min_day, max_day, prefix, suffix):
    image_names = list_of_image_names(min_well, max_well, min_day, max_day, prefix, suffix)
    images = []
    for name in image_names:
        img = Image.open(name)
        images.append(img)

    return images

def get_single_images_tensor(image_names):
    listTens = []
    for name in image_names:
        img = Image.open(name).convert('RGB')
        t_img = Variable(to_tensor(scaler(img)))
        listTens.append(t_img)

    return torch.stack(listTens)

def get_3D_tensors(image_names, num_stack):
    listTens = []
    count = 0
    currT = []
    for name in image_names:
        img = Image.open(name).convert('RGB')
        t_img = Variable(to_tensor(scaler(img)))
        count += 1
        currT.append(t_img)
        if count != 0 and count % num_stack == 0:
            tenS = torch.stack(currT, dim = 1)
            #tenS = tenS.squeeze(0)
            listTens.append(tenS)
            currT = []
            count = 0

    return torch.stack(listTens)

# abstractions that help us generate the image tensors
class Single_Image_Loader:
    def __init__(self, day, all_well_ids, prefix, suffix):
        self.image_names = list_of_image_names_from_set_of_wells(all_well_ids, day, day, prefix, suffix)
        self.image_tensor = get_single_images_tensor(self.image_names)

    def get_image_tensor(self):
        return self.image_tensor

class TimeCourse_Image_Loader:
    def __init__(self, min_day, max_day, all_well_ids, prefix, suffix):
        self.image_names = list_of_image_names_from_set_of_wells(all_well_ids, min_day, max_day, prefix, suffix)
        self.image_tensor = get_3D_tensors(self.image_names, (max_day - min_day + 1))

    def get_image_tensor(self):
        return self.image_tensor
