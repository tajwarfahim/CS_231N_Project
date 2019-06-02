import numpy as np
import pandas as pd

def read_excel_file(file_name):
    data_frame = pd.read_excel(file_name)
    return data_frame

def read_labels(data_frame):
    num_rows, num_cols = data_frame.shape
    map_of_labels = {}

    for i in range(0, num_rows):
        well_name = data_frame.iloc[i, 0]
        well_id = int(well_name[4:])
        map_of_labels[well_id] = data_frame.iloc[i, 1]

    return map_of_labels

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

def create_label_vector(label_map, label_to_label_id):
    max_well_id = get_max_key(label_map)
    label_vector = []

    for well_id in range(0, max_well_id):
        label = label_map[well_id]
        label_id = label_to_label_id[label]
        label_vector.append(label_id)

    return label_vector

class Label_Reader:
    def __init__(self, file_name):
        data_frame = read_excel_file(file_name)
        self.label_map = read_labels(data_frame)
        self.type_of_labels, self.set_of_labels = find_type_of_labels(self.label_map)
        self.label_to_label_id, self.label_id_to_label = enumerate_labels(self.set_of_labels)
        self.label_vector = create_label_vector(self.label_map, self.label_to_label_id)

    def get_label_vector(self):
        return self.label_vector

    def get_label_map(self):
        return self.label_map

    def get_label_id_to_label_map(self):
        return self.label_id_to_label

    def get_number_of_different_labels(self):
        return self.type_of_labels

    def get_all_labels(self):
        return self.set_of_labels
