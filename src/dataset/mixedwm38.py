import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os
from PIL import Image
import cv2


import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader


class MixedWM38(Dataset):
    def __init__(self, path='data/MixedWM38.npz', train=True, val=False, transforms=None,
                 input_channels=1, num_classes=8, train_size=0.6, random_state=0):
        super().__init__()
        self.path = os.path.expanduser(path)
        self.train = train
        self.val = val
        self.transforms = transforms
        self.input_channels = input_channels
        self.num_classes = num_classes
        
        self.X, self.y = self.load_data(random_state=random_state, train_size=train_size)
        
    def load_data(self, random_state, train_size=0.6):
        data = np.load(self.path)
        wafers = data['arr_0']
        labels = data['arr_1']
        X_train, X_test, y_train, y_test = train_test_split(wafers, labels, train_size=train_size, 
                                                            shuffle=True, random_state=random_state)
        
        if self.train:
            return X_train, y_train
        
        
        X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size = 0.5, 
                                                        shuffle=True, random_state=random_state)
        
        if self.val:
            return X_val, y_val
        
        return X_test, y_test
    
    def set_transform(self, transforms):
        self.transforms = transforms
    
    def __len__(self):
        return self.X.shape[0]
        
    def bin_img(self, img, lwr_thre=128, upr_thre=255):
        img[img <= lwr_thre] = 0
        
        return img
    
    def __getitem__(self, index):
        wafer = self.X[index].astype('float32')
        label = self.y[index]
        wafer /= 2
        wafer *= 255
        wafer = self.bin_img(wafer)  
        wafer = torch.from_numpy(wafer).unsqueeze(0).to(dtype=torch.uint8)
        
        if self.input_channels == 3:
            wafer = wafer.expand(3, -1, -1)
        
        if self.transforms != None:
            wafer = torchvision.transforms.ToPILImage()(wafer)
            wafer = self.transforms(wafer)
        
        label = torch.from_numpy(label).to(dtype=torch.float)
        
        return wafer, label