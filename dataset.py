import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from utils_cells import get_images_list, transform_image, transform_target, resize_with_padding
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import numpy as np
import torchvision.transforms.functional as F
import torch
from torchvision import transforms
from torchvision.transforms import functional as F
import cv2
from sklearn.model_selection import train_test_split


class ImageDataset(Dataset):
    def __init__(self, data_path, transform=None, target_transform=None, reduce=False):
        self.transform = transform
        self.target_transform = target_transform
        self.dataset = shuffle(self.load_dataset(data_path))
        if reduce:
            self.__remove_small_images()
            #self.dataset = self.dataset.reset_index()
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

    def __remove_small_images(self):
        for i in range(len(self.dataset)-1):
            image = cv2.imread(f'{self.dataset["filename"].loc[i]}')
            if image.shape[0] < 12 or image.shape[1] < 12:
                self.dataset = self.dataset.drop(i)
        self.dataset = self.dataset.reset_index()
    
    def __getitem__(self, idx):
        image = cv2.imread(f'{self.dataset["filename"].loc[idx]}')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #image = cv2.resize(image, (32, 32), interpolation=cv2.INTER_CUBIC)
        image = resize_with_padding(image, (32, 32))
        image = image.astype(np.float32)
        image = self.transform(image = image)['image'] if self.transform is not None else image

        target = self.dataset["class"].loc[idx]

        if target == 'normal.':
            target_ = [1, 0, 0, 0]
        elif target == 'inflamatory.':
            target_ = [0, 1, 0, 0]
        elif target == 'tumor.':
            target_ = [0, 0, 1, 0]
        elif target == 'other.':
            target_ = [0, 0, 0, 1]
        else:
            print(target)
        
        image = F.to_tensor(image)
        
       
     

        """To see transorms use:
            image, target = trainset[15]
            image = image.numpy()
            image=np.swapaxes(image,0,1)
            image=np.swapaxes(image,1,2)
            plt.imshow(image)"""

        return image.float(), torch.Tensor(np.array(target_, dtype=np.float32))



