import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import cv2


import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split

from torchvision import transforms


class MMACDataSet(Dataset):
    def __init__(self, root, train=True, transform=None):
        
        if train:
            anno_PATH = root+'train.csv'
            img_PATH = root+'train/'
        else:
            anno_PATH = root+'valid.csv'
            img_PATH = root+'valid/'
            
        df = pd.read_csv(anno_PATH)
        
        labels = df['spherical_equivalent'].values
        image_names = df['image'].values
        # Non-image data
        self.age = df['age'].values
        self.sex = df['sex'].apply(lambda x: 1 if x == 'male' else 0).values  # Convert 'male'/'female' to 1/0
        self.height = df['height'].values
        self.weight = df['weight'].values
        
        self.anno_PATH = anno_PATH
        self.img_PATH = img_PATH
        self.image_names = image_names
        self.labels = labels
        
        self.train = train
        
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(size=(512, 512),
                                  interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize([0.386, 0.186, 0.024], 
                                     [0.241, 0.125, 0.049])
                ])
        else:
            self.transform = transform
        
    def __getitem__(self, index):
        '''Take the index of item and returns the image and its labels'''
        label = self.labels[index]
        
        image_name = self.image_names[index]
        image = cv2.imread(self.img_PATH+image_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        image = self.transform(image)
        # Collect the non-image data for the current index
        age = self.age[index]
        sex = self.sex[index]
        height = self.height[index]
        weight = self.weight[index]
        patient_info = torch.tensor([age, sex, height, weight], dtype=torch.float32)
        
        return image, patient_info, torch.FloatTensor([label])
    
    def __len__(self):
        return len(self.image_names)
    
    
    
if __name__ == '__main__':
    
    root = ''
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop((512, 512), scale=(0.08, 1.0), 
                                     ratio=(0.75, 1.35), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=[-180, 180],
                                fill=0,interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ColorJitter(brightness=0.04, contrast=0.04, saturation=0.04, hue=0.04),
        transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
        transforms.RandomEqualize(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.386, 0.186, 0.024], 
                             [0.241, 0.125, 0.049]),
        transforms.RandomErasing(p=0.25)
    ])
    
    
    dataset = MMACDataSet(root, transform=transform)
        
    train_loader = DataLoader(
        dataset, batch_size=128, shuffle=False, drop_last=False, num_workers=0
    )
    
    mean = np.zeros(3)
    std = np.zeros(3)
    total_images = 0
    
    images = []
    for idx, data in enumerate(train_loader):
        image, target = data
        #images.append(image)

        print(target.shape)
    
    #images = torch.cat(images, dim=0)
    
    
    #print("Mean:", np.mean(images.numpy(), axis=(0, 2, 3)))
    #print("Std:", np.std(images.numpy(), axis=(0, 2, 3)))

        
        
        
        
        
        
        
        
        
        
        
        

        
        
        
