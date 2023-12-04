"""
Splits images into test_set and dataset without test_set
Script saves paths to images in txt files. 
"""
import os
import glob
import random
from utils_cells import split_data

random.seed(0)

def save_images(files, dataset_type, class_): 
    log = open(f'{dataset_type}/data_{class_}.txt', 'a')
    for file in files:
        log.write(f'{file}\n')
    log.close()

def set_data_without_test(dataset, test_dataset):
    for category, test_category in zip(dataset, test_dataset):
        for file in test_category:
            if file in category:
                category.remove(file)
    return dataset
    

classes = ['inflamatory', 'normal', 'tumor', 'other']
inflamatory = glob.glob('cells_final/inflammatory/*')
normal = glob.glob('cells_final/normal/*')
tumor = glob.glob('cells_final/tumor/*')
other = glob.glob('cells_final/other/*')

inflamatory_pannuke = random.sample([file for file in inflamatory if "PanNuke" in file], 500)
inflamatory_monusac = random.sample([file for file in inflamatory if "MoNuSAC" in file], 500)
other_test = random.sample(other, 100)
tumor_test = random.sample(tumor, 1000)
normal_test = random.sample(normal, 1000)


if len(os.listdir('test_data')) > 0:
    for file in os.listdir('test_data'):
        os.remove(f'test_data/{file}')
save_images(inflamatory_pannuke, 'test_data','inflamatory_test')
save_images(inflamatory_monusac, 'test_data','inflamatory_test')
save_images(other_test, 'test_data','other_test')
save_images(tumor_test, 'test_data','tumor_test')
save_images(normal_test, 'test_data','normal_test')


inflamatory_monusac.extend(inflamatory_pannuke)
dataset = set_data_without_test(dataset=[inflamatory,normal,tumor,other], test_dataset=[inflamatory_monusac, normal_test, tumor_test, other_test])
if len(os.listdir('data_without_test')) > 0:
    for file in os.listdir('data_without_test'):
        os.remove(f'data_without_test/{file}')
for data, cls in zip(dataset, classes):
    save_images(data, 'data_without_test', cls)

dataset_train, dataset_validation = split_data(path='data_without_test', train_size=0.7)

if len(os.listdir('train_data')) > 0: 
    for file in os.listdir('train_data'):
        os.remove(f'train_data/{file}')

for data, cls in zip(dataset_train, classes):
    save_images(data, 'train_data', cls)

if len(os.listdir('validation_data')) > 0: 
    for file in os.listdir('validation_data'):
        os.remove(f'validation_data/{file}')

for data, cls in zip(dataset_validation, classes): 
    save_images(data, 'validation_data', cls)

