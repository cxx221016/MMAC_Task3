import time
import pickle
import random
import csv
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
import torch.optim.lr_scheduler as lr_scheduler


from torch_optimizer import AdamP

from model import ResNet50
from materials import MMACDataSet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def mixup_data(x, y, alpha=1.0):
    batch_size = x.size()[0]
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(batch_size)
    mixed_x = lam*x + (1-lam)*x[index,:]
    mixed_y = lam*y + (1-lam)*y[index]
    
    return mixed_x, mixed_y

def train(train_loader, net, optimizer, epoch, scheduler):
    batch_time = AverageMeter()
    data_time  = AverageMeter()
    losses     = AverageMeter()
    
    net.train()
    end = time.time()
    
    for idx, data in enumerate(train_loader):
        data_time.update(time.time() - end)
        img, target = data
        
        img, target = mixup_data(img, target, alpha=1.0)
        
        img = img.to(device)
        target = target.to(device)
        
        output = net(img)
        loss =  nn.L1Loss()(output, target)
        optimizer.zero_grad()
        
        loss.backward()
        optimizer.step()
        
        losses.update(loss.detach().item(), img.size(0))
        
        batch_time.update(time.time() - end)
        end = time.time()
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        print('Epoch: [{0}][{1}/{2}]\t'
              'Time {batch_time.val:.3f} ({batch_time.sum:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.sum:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Learning Rate: {current_lr}\t'.format(
               epoch, idx, len(train_loader), batch_time=batch_time,
               data_time=data_time, loss=losses, current_lr = current_lr))
        
    return losses.avg


def validate(val_loader, net):
    losses = AverageMeter()
    
    net.eval()
    
    with torch.no_grad():
        for idx, data in enumerate(val_loader):
            img, target = data
            img = img.to(device)
            target = target.to(device)
            
            output = net(img)
            loss =  nn.L1Loss()(output, target)
            
            losses.update(loss.detach().item(), img.size(0))
            
    print('Testing: Loss ({loss.avg:.4f})'.format(loss=losses))

    return losses.avg


if __name__ == '__main__':
    
    root = './Prediction of Spherical Equivalent/'
    ckpt = './weights/ReXNetV2.pth'

    batch_size = 96
    lr = 1e-3
    epochs = 800
    
    lr_warmup_decay = 1e-5
    eta_min = 1e-6
    warmup_epoch = 20
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(size=(512, 512),
                          interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=[-180, 180],
                                fill=0,interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ColorJitter(brightness=0.04, contrast=0.04, saturation=0.04, hue=0.04),
        transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.386, 0.186, 0.024], 
                             [0.241, 0.125, 0.049]),
        transforms.RandomErasing(p=0.25)
    ])
    
    train_dataset = MMACDataSet(root, train=True, transform=transform)
    val_dataset = MMACDataSet(root, train=True, transform=None)
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  drop_last=False, num_workers=8)
    
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                num_workers=0)
    
    net = ResNet50(1)
    #net = ReXNetV2(width_mult=1.0, classes=1)
    net = nn.DataParallel(net).to(device)
    #cudnn.benchmark = True   
    
    optimizer = AdamP(net.parameters(), lr = lr, weight_decay = 0.001) 
    
    steps_per_epoch = len(train_dataloader)
    
    warmup_lr_scheduler =  lr_scheduler.LinearLR(optimizer, start_factor=lr_warmup_decay, total_iters = warmup_epoch * steps_per_epoch)
    main_lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max= (epochs-warmup_epoch) * steps_per_epoch, eta_min=eta_min)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer, schedulers=[warmup_lr_scheduler, main_lr_scheduler], milestones=[warmup_epoch * steps_per_epoch]
            )
    
    print('train')
    start_time = time.time()
    best_val = 1e10
    for epoch in range(epochs):
       train_logs = train(train_dataloader, net, optimizer, epoch, scheduler)
       
       
       if (epoch+1)%10 == 0:
           val_logs = validate(val_dataloader, net)  
           torch.save(net.module.state_dict(), ckpt)
           
    print('%d epochs training and val time : %.2f'%(epochs, time.time()-start_time))
    
    
    
    
    
    
    
    
    
    
    
    
    
        
        
        
        
        
        
        
        
