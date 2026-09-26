import os
import torch
import torchvision

from dataset.mixedwm38 import MixedWM38
from dataset.wm811k import WM811K


class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]
    

def load_dataset(
    data_name='mixedwm38', 
    path="./data/MixedWM38.npz", 
    train=True, val=False, calib=False, 
    supcon=False, 
    input_channels=1, img_size=(64, 64), 
    num_classes=8, train_size=0.6, random_state=0, idx=None
):
    
    if train or val:
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(img_size),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomVerticalFlip(),
            torchvision.transforms.RandomRotation(90),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, ) * input_channels, (0.5, ) * input_channels)
        ]) 
    else:
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(img_size),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, ) * input_channels, (0.5, ) * input_channels)
        ]) 
            
    if data_name == 'mixedwm38':
        data = MixedWM38(
            path=path, 
            train=train, val=val, transforms=transforms, 
            input_channels=input_channels, num_classes=num_classes,
            train_size=train_size, random_state=random_state
        )
    elif data_name == 'wm811k':
        data = WM811K(
            path=path, train=train, val=val, transforms=transforms, 
            input_channels=input_channels, num_classes=num_classes, idx=idx
        )
    
    return data


def get_loaders(train_dataset, test_dataset, val_dataset=None, batch_size=128):
    
    if train_dataset != None:
        train_loader = torch.utils.data.DataLoader(train_dataset, 
                                                   batch_size=batch_size,
                                                   shuffle=True, 
                                                   num_workers=2) 
    else:
        train_loader = None

    if test_dataset != None:
        test_loader = torch.utils.data.DataLoader(test_dataset, 
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  num_workers=2)
    else:
        test_loader = None
        
    if val_dataset != None:
        val_loader = torch.utils.data.DataLoader(val_dataset, 
                                                 batch_size=batch_size,
                                                 shuffle=True,
                                                 num_workers=2)
    else:
        val_loader = None
        

    return train_loader, test_loader, val_loader