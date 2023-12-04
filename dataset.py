import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from utils_cells import get_images_list, transform_image, transform_target
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

class ImageDataset(Dataset):
    def __init__(self, data_path, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.dataset = shuffle(self.load_dataset(data_path))

    def load_dataset(self, path):
        files = os.listdir(path)
        dataset_final = pd.DataFrame()
        dataset_final['filename'] = []
        dataset_final['class'] = []
        for filename in files:
            dataset = pd.DataFrame()
            if filename.endswith('.txt'):
                files = get_images_list(f'{path}/{filename}')
                dataset['filename'] = files
                dataset['class'] = filename.split('_')[1][:-3]
                dataset_final = pd.concat([dataset_final, dataset], ignore_index=True)
        return dataset_final                
                          
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image = plt.imread(f'{self.dataset["filename"].loc[idx]}')
        image = self.transform(image) if self.transform is not None else image
        target = self.target_transform(self.dataset['class'][idx]) if self.target_transform is not None else self.dataset['class'][idx]

        return image, target







