# author: Fahim Tajwar
# help taken from : https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

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

# helper functions
def get_accuracy_per_class(correct_map, label_id_to_label, frequency_map_per_class):
    accuracy_map = {}
    correct_map_per_class = {}

    for key in correct_map:
        correct_map_per_class[label_id_to_label[key]] = correct_map[key]
        accuracy_map[label_id_to_label[key]] = float(correct_map[key]) / frequency_map_per_class[label_id_to_label[key]]

    return accuracy_map, correct_map_per_class

def test_model(model, test_loader, label_id_to_label_map, verbose = True, device = 'cpu'):
    correct = 0
    total = 0
    correct_map = {}
    frequency_map_per_class  = {}

    for images, labels in test_loader:
        images = Variable(images).to(device)
        output = model(images).to(device)
        _, predicted = torch.max(output.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

        if label_id_to_label_map[labels[0].item()] in frequency_map_per_class:
            frequency_map_per_class[label_id_to_label_map[labels[0].item()]] += 1
        else:
            frequency_map_per_class[label_id_to_label_map[labels[0].item()]] = 1

        if labels[0].item() not in correct_map:
            if predicted[0] == labels[0]:
                correct_map[labels[0].item()] = 1
            else:
                correct_map[labels[0].item()] = 0

        else:
            if predicted[0] == labels[0]:
                correct_map[labels[0].item()] += 1

    accuracy = (100.0 * correct) / total
    accuracy_per_class_map, correct_map_per_class_map = get_accuracy_per_class(correct_map, label_id_to_label_map, frequency_map_per_class)

    if verbose:
        print("accuracy : %f" % accuracy)

        print("")
        print("Frequency per class: ", frequency_map_per_class)
        plot_bar_graph_from_map(frequency_map_per_class, "Class", "Frequency", [key for key in frequency_map_per_class.keys()])

        print("Number of datapoints we got correct per class", correct_map_per_class_map)
        print("Accuracy per class", accuracy_per_class_map)
        plot_bar_graph_from_map(accuracy_per_class_map, "Class", "accuracy", [key for key in accuracy_per_class_map.keys()])

    # the multiple of all accuracies, we use this our evaluation metric
    composite_accuracy = 1.0
    for image_class in accuracy_per_class_map:
        composite_accuracy *= accuracy_per_class_map[image_class]

    return composite_accuracy

# abstract class model to train and test our different models
class Model:
    def __init__(self, model, training_set, batch_size, learning_rate, label_id_to_label_map, weight = None,
                imbalanced_class = False, num_epochs = 25, verbose = True, device = "cpu"):
        self.model = model
        self.training_set = training_set
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.verbose = verbose
        self.label_id_to_label_map = label_id_to_label_map
        self.imbalanced_class = imbalanced_class
        self.weight = weight

    def train(self, device = "cpu"):
        model, learning_rate, num_epochs, verbose = self.model, self.learning_rate, self.num_epochs, self.verbose
        train_loader = torch.utils.data.DataLoader(dataset = self.training_set, batch_size = self.batch_size, shuffle=True)

        criterion = nn.CrossEntropyLoss()
        if self.imbalanced_class:
            criterion = nn.CrossEntropyLoss(self.weight)

        optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)
        loss_history = []

        for epoch in range(num_epochs):
            running_loss = 0
            num_iters = 0
            for i, (images, labels) in enumerate(train_loader):
                images = Variable(images).to(device)
                labels = Variable(labels).to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.data
                num_iters += 1

            average_loss = float(running_loss) / num_iters
            if verbose:
                print ('Epoch: [%d/%d], Loss: %.4f' % (epoch+1, num_epochs, average_loss))

            loss_history.append(average_loss)

        print("Training done!")
        self.model = model
        if verbose:
            x_axis = range(num_epochs)
            y_axis = loss_history
            plot_x_vs_y(x_axis, y_axis, x_title = "Training epoch", y_title= "Average Training Loss")
            train_loader_testing = torch.utils.data.DataLoader(dataset = self.training_set,
                                               batch_size = 1,
                                               shuffle=True)
            return self.test(train_loader_testing, verbose, device)


    def test(self, test_loader, verbose = True, device = "cpu"):
        return test_model(self.model, test_loader, self.label_id_to_label_map, verbose, device)


# our abstract class to do the cross validation for us
# note that we only can manipulate the hyperparameter learning rate in our current setting

class Hyperparameter_Tuner:
    def __init__(self, given_model, training_set, validation_loader, batch_size, learning_rates, label_id_to_label_map,
                weight = None, imbalanced_class = False, num_epochs = 10, device = "cpu"):

        assert(len(learning_rates) > 0)

        best_model = None
        best_validation_accuracy = float('-inf')
        for lr in learning_rates:
            model = Model(given_model, training_set, batch_size, lr, label_id_to_label_map, weight = weight,
                        imbalanced_class  = imbalanced_class, num_epochs = num_epochs, verbose = False)
            model.train()
            validation_accuracy = model.test(validation_loader, verbose = False)

            if validation_accuracy > best_validation_accuracy:
                best_validation_accuracy = validation_accuracy
                best_model = model

        self.best_model = best_model

    def get_best_model(self):
        return self.best_model
