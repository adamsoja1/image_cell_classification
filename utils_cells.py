"""
Functions and tools for further analysis, implemenation etc.
"""
import os
import glob
import random
import cv2
from sklearn.metrics import confusion_matrix, classification_report
import torch
import numpy as np

random.seed(0)

def get_images_list(path):
    """Reads list of images from a txt file."""
    file = open(path, 'r')
    images = []
    for line in file:
        images.append(line.strip())
    return images

def split_data(path, train_size):
    """Splits data into training and validation sets."""
    inflamatory = get_images_list(f'{path}/data_inflamatory.txt')
    normal = get_images_list(f'{path}/data_normal.txt')
    other = get_images_list(f'{path}/data_other.txt')
    tumor = get_images_list(f'{path}/data_tumor.txt')

    random.shuffle(inflamatory)
    random.shuffle(normal)
    random.shuffle(other)
    random.shuffle(tumor)

    train_inflamatory = inflamatory[:int(train_size*len(inflamatory))]
    train_normal = normal[:int(train_size*len(normal))]
    train_other = other[:int(train_size*len(other))]
    train_tumor = tumor[:int(train_size*len(tumor))]

    validation_inflamatory = inflamatory[int(train_size*len(inflamatory)):]
    validation_normal = normal[int(train_size*len(normal)):]
    validation_other = other[int(train_size*len(other)):]
    validation_tumor = tumor[int(train_size*len(tumor)):]

    dataset_train = [train_inflamatory, train_normal,train_tumor, train_other]
    dataset_validation = [validation_inflamatory, validation_normal, validation_tumor, validation_other]
    return dataset_train, dataset_validation

def transform_target(target):
    if target == 'inflamatory':
        return [1, 0, 0, 0]
    elif target == 'normal':
        return [0, 1, 0, 0]
    elif target == 'tumor':
        return [0, 0, 1, 0]
    elif target == 'other':
        return [0, 0, 0, 1]
    

def transform_image(image):
    image = cv2.resize(image, (32, 32))
    return image


def test_report(model, dataloader):
    """Prints confusion matrix for testing dataset
    dataloader should be of batch_size=1."""

    y_pred = []
    y_test = []
    model.eval()
    with torch.no_grad():
        for data, label in dataloader:
            output = model(data)
            label = label.numpy()
            output = output.numpy()

          

            y_pred.append(np.argmax(output))
            y_test.append(np.argmax(label))
        print(y_pred)
        print(y_test)
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred))





