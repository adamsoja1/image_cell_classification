"""
Functions and tools for further analysis, implemenation etc.
"""
import os
import glob
import random
import cv2
from sklearn.metrics import precision_score, recall_score

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




def calculate_precision_recall_per_class(y_true, y_pred, num_classes=4):
    """
    Calculate precision and recall per class for a classification task with 4 classes.

    Parameters:
    - y_true: true class labels
    - y_pred: predicted class labels
    - num_classes: total number of classes

    Returns:
    - precision_per_class: a list containing precision for each class
    - recall_per_class: a list containing recall for each class
    """

    precision_per_class = []
    recall_per_class = []

    for class_label in range(num_classes):
        # Create binary labels for the current class
        true_class = (y_true == class_label)
        pred_class = (y_pred == class_label)

        # Calculate precision and recall for the current class
        precision = precision_score(true_class, pred_class, zero_division=0)
        recall = recall_score(true_class, pred_class, zero_division=0)

        # Append precision and recall to the respective lists
        precision_per_class.append(precision)
        recall_per_class.append(recall)

    return precision_per_class, recall_per_class



def get_accuracies_per_class(resutls):
    return results[0], results[1], results[2], results[3]






