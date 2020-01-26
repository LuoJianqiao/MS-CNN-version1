#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 16:46:42 2019

@author: a
"""

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
from torchvision import datasets, models, transforms
import time
import os
import copy


###########data set######################
#the parameters in the normalize operation is fot the NR dataset. 
#if you change the dataset, please re-compute the means and variances for the dataset

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.37, 0.38, 0.34], [0.14, 0.13, 0.13])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.37, 0.38, 0.34], [0.14, 0.13, 0.13])
    ])}

data_dir='../data'

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train','val']}

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=16,
                                             shuffle=True, num_workers=4)
              for x in ['train','val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train','val']}
class_names = image_datasets['train'].classes
num_classes=len(class_names)
###########data set######################

#############model######################
model=models.vgg16_bn(pretrained=True)
num_ftrs=model.classifier[6].in_features
model.classifier[6]=nn.Linear(num_ftrs,num_classes)
model = model.cuda()
#############model######################

############criteria ,optimizer, learn rate####################
criterion = nn.CrossEntropyLoss()
opt = optim.SGD([{'params':model.features.parameters()},
                 {'params':model.classifier.parameters(),'lr':0.01}]
                , lr=0.001, momentum=0.9,weight_decay=5e-4)
lr_v = lr_scheduler.ReduceLROnPlateau(opt, mode='min',factor=0.1,patience=3,verbose=True)
############criteria ,optimizer, learn rate####################

############para initialise####################
wts_val=copy.deepcopy(model.state_dict())
num_epochs=100
############para initialise####################

#########record and result##################    
loss_history=np.zeros((num_epochs,))
acc_history=np.zeros((num_epochs,))
Sn=5# the checkpoint accounting the 5th highest accuracy on the validation is regarded as the final model
result_dir='./train_hard'# path to save the training results, including accuracy curve, loss curve, and weights
if not os.path.exists(result_dir):
    os.mkdir(result_dir)    
#########record and result################## 

for epoch in range(num_epochs):
    
    print('ep0: {}/{}'.format(epoch+1,num_epochs))
    print('phase: train')
    model.train()
    epo_loss=0.0    
    since=time.time()
    ####################train########################
    for inputs,labels in dataloaders['train']:
      
        inputs=inputs.cuda()
        labels=labels.cuda()
        opt.zero_grad() 
        
        outputs=model(inputs)
        loss=criterion(outputs,labels)      
        loss.backward()
        opt.step()
        
        epo_loss+=loss.item()*inputs.size(0)
    epo_loss/=dataset_sizes['train']
    
    lr_v.step(epo_loss)
    loss_history[epoch,]=epo_loss
    
    end_time1=time.time()
    print('loss is {}'.format(epo_loss))
    print('training time is {:3f} s'.format(end_time1-since))
        
    ########################val##########################
    print('phase: validate')
    model.eval()
    running_corrects = 0.0
    for inputs,labels in dataloaders['val']:
        inputs=inputs.cuda()
        labels=labels.cuda()
        outputs=model(inputs)
        _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds == labels.data)
    
    epoch_acc = running_corrects.double() / dataset_sizes['val']
    acc=epoch_acc.cpu().data.numpy()
    acc_history[epoch,]=acc
    if epoch<Sn:
        wts_path=result_dir+'/w_'+str(epoch)+'.pth'
        wts_val=copy.deepcopy(model.state_dict())
        torch.save(wts_val,wts_path)
        acc_order=np.sort(acc_history)
        acc_order1=acc_order[-Sn:num_epochs]
        
    if epoch>=20:   
        if acc>acc_order1[0]:
            qi=Sn-1
            while acc<acc_order1[qi]:
                qi-=1
            
            for pi in range(qi+1):
                old_path=result_dir+'/w_'+str(pi)+'.pth'
                new_path=result_dir+'/w_'+str(pi-1)+'.pth'
                os.rename(old_path,new_path)
            
            save_path=result_dir+'/w_'+str(pi)+'.pth'
            wts_val=copy.deepcopy(model.state_dict())
            torch.save(wts_val,save_path)
            
            acc_order1=np.insert(acc_order1,qi+1,acc)
            acc_order1=np.delete(acc_order1,0)
 
    end_time2=time.time()    
    print('val Acc: {:4f}'.format(epoch_acc))
    print('validate time is {:3f} s'.format(end_time2-end_time1))
    print('\n\n')
    
    
np.save(result_dir+'/loss.npy',loss_history)
np.save(result_dir+'/acc.npy',acc_history)
np.save(result_dir+'/top_acc.npy',acc_order1)
    
    

    
    
    
    
    
    
    