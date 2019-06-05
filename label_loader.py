# Author: Fahim Tajwar
# this was in total implemented by the author
# allows us to modify which classes we want to include in our dataset

import numpy as np
import pandas as pd
import torch
from util import *

# constants

#ALLOWED_LABELS = ['cell dies', 'grows sparse', 'grows dense', 'transient cells', 'edge artifact', 'debris']
ALLOWED_LABELS = ['cell dies', 'grows sparse', 'grows dense']
#ALLOWED_LABELS = ['cell dies', 'grows sparse']

LABEL_CONVERSION_MAP = {'cell dies': 'cell dies',
                        'dies': 'cell dies',
                        'grows sparse': 'grows sparse',
                        'sparse': 'grows sparse',
                        'grows dense' : 'grows dense',
                        'dense': 'grows dense',
                        'transient cells': 'transient cells',
                        'edge artifact': 'edge artifact',
                        'debris' : 'debris'
                        }

# helper functions
def read_excel_file(file_name):
    data_frame = pd.read_excel(file_name)
    return data_frame

def get_well_id_from_well_name(well_name):
    well_name = str(well_name)
    assert(len(well_name) > 0)
    if well_name[0] == 'w':
        return int(well_name[4:])
    else:
        return int(well_name)

def get_well_label_from_given_label(given_label):
    if given_label == 'sparse' or given_label == 'grows sparse':
        return 'grows sparse'
    elif given_label == 'grows dense' or given_label == 'dense':
        return 'grows dense'
    elif given_label == 'cell dies' or given_label == 'dies':
        return 'cell dies'
    else:
        return given_label

def read_labels(file_names):
    label_map = {}
    all_well_ids = []

    for file_name in file_names:
        data_frame = read_excel_file(file_name)
        num_rows, num_cols = data_frame.shape
        for i in range(num_rows):
            well_id = get_well_id_from_well_name(data_frame.iloc[i, 0])
            well_label = get_well_label_from_given_label(data_frame.iloc[i, 1])
            if well_label in ALLOWED_LABELS:
                label_map[well_id] = well_label
                all_well_ids.append(well_id)

    return label_map, all_well_ids

def find_type_of_labels(label_map):
    set_of_labels = set()
    num_types = 0

    for well_id in label_map:
        if label_map[well_id] not in set_of_labels:
            num_types += 1
            set_of_labels.add(label_map[well_id])

    return num_types, set_of_labels

def enumerate_labels(set_of_labels):
    label_to_label_id = {}
    label_id_to_label = {}

    num = 0
    for label in set_of_labels:
        label_to_label_id[label] = num
        label_id_to_label[num] = label
        num += 1

    return label_to_label_id, label_id_to_label


def get_max_key(dictionary):
    maximum = float('-inf')
    for key in dictionary:
        if key > maximum:
            maximum = key

    return maximum

def create_label_vector(label_map, label_to_label_id, all_well_ids):
    label_vector = []
    for well_id in all_well_ids:
        label = label_map[well_id]
        label_id = label_to_label_id[label]
        label_vector.append(label_id)

    return label_vector

def get_class_distribution(label_id_to_label_map, label_vector):
    class_distribution = {}
    for i in range(len(label_vector)):
        label = label_vector[i]
        if label_id_to_label_map[label] in class_distribution:
            class_distribution[label_id_to_label_map[label]] += 1
        else:
            class_distribution[label_id_to_label_map[label]] = 1

    return class_distribution

def get_max_val(dictionary):
    maximum = float('-inf')
    for i in dictionary:
        if i > maximum:
            maximum = i

    return maximum

def calculate_weight_vector(label_vector):
    max_label = max(label_vector)
    weight_vector = [i for i in range(max_label + 1)]
    map = {}
    for i in label_vector:
        if i not in map:
            map[i] = 1
        else:
            map[i] += 1

    max_val = get_max_val(map)
    for i in map:
        weight_vector[i] = float(max_val) / map[i]

    return torch.tensor(weight_vector)

# the abstract class that takes care of these things for us
class Label_Reader:
    def __init__(self, file_names):
        self.label_map, self.all_well_ids = read_labels(file_names)
        self.type_of_labels, self.set_of_labels = find_type_of_labels(self.label_map)
        self.label_to_label_id, self.label_id_to_label = enumerate_labels(self.set_of_labels)
        self.label_vector = create_label_vector(self.label_map, self.label_to_label_id, self.all_well_ids)
        self.num_data_points = len(self.label_map)
        self.class_distribution = get_class_distribution(self.label_id_to_label, self.label_vector)
        self.weight_vector = calculate_weight_vector(self.label_vector)

    def get_label_vector(self):
        return self.label_vector

    def get_label_map(self):
        return self.label_map

    def get_label_id_to_label_map(self):
        return self.label_id_to_label

    def get_label_to_label_id_map(self):
        return self.label_to_label_id

    def get_number_of_different_labels(self):
        return self.type_of_labels

    def get_all_labels(self):
        return self.set_of_labels

    def get_number_of_data_points(self):
        return self.num_data_points

    def get_class_distribution(self):
        return self.class_distribution

    def show_class_disribution_histogram(self):
        label_for_each_class = list(self.class_distribution.keys())
        plot_bar_graph_from_map(self.class_distribution, "Classes", "Frequency", label_for_each_class)

    def get_all_well_ids(self):
        return self.all_well_ids

    def get_weight_vector(self):
        return self.weight_vector
