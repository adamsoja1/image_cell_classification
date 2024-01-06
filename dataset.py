import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from utils_cells import get_images_list, transform_image, transform_target
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import numpy as np
import torchvision.transforms.functional as F
import torch
from torchvision import transforms
from torchvision.transforms import functional as F



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
        image = image/image.max()
        image = self.transform(image = image)['image'] if self.transform is not None else image

        target = self.dataset["class"].loc[idx]

        if target == 'inflamatory.':
            target_ = [1, 0, 0, 0]
        elif target == 'normal.':
            target_ = [0, 1, 0, 0]
        elif target == 'other.':
            target_ = [0, 0, 1, 0]
        elif target == 'tumor.':
            target_ = [0, 0, 0, 1]
        else:
            print(target)
        
        image = F.to_tensor(image)
        
        if self.transform is None:
            image = F.resize(image, (32, 32))
     

        """To see transorms use:
            image, target = trainset[15]
            image = image.numpy()
            image=np.swapaxes(image,0,1)
            image=np.swapaxes(image,1,2)
            plt.imshow(image)"""

        return image, torch.Tensor(np.array(target_, dtype=np.float32))



