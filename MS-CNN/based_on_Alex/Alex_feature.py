#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 25 16:36:26 2019

@author: a
"""

from __future__ import print_function, division
import os
import torch
from torchvision import datasets, models, transforms
from PIL import Image 
import pickle
import torch.nn as nn


###########data set######################
train_dir='../data/train'
train_datasets=datasets.ImageFolder(train_dir)
class_names=train_datasets.classes
num_classes=len(class_names)

#the parameters in the normalize operation is fot the NR dataset. 
#if you change the dataset, please re-compute the means and variances for the dataset
trans=transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.37, 0.38, 0.34], [0.14, 0.13, 0.13])])
###########data set######################

#############model and load weight######################
model=models.alexnet(pretrained=False)
num_ftrs=model.classifier[6].in_features
model.classifier[6]=torch.nn.Linear(num_ftrs,num_classes)
model = model.cuda()
model.eval()
wts=torch.load('train_hard/w_0.pth')
model.load_state_dict(wts)
#############model and load weight######################

##############define hook###########################
modules=list(model.features._modules.items())
forward_outputs=[]
def forward_hook_function(module,ten_in,ten_out):
    forward_outputs.append(ten_out)

modules[5][1].register_forward_hook(forward_hook_function)   #record the middle-level feature 
modules[12][1].register_forward_hook(forward_hook_function) #record the high-level feature
##############define hook###########################

############pool function################
pool=nn.MaxPool2d(kernel_size=2,stride=2)
############pool function################

###########target directory################
train_dir='../data/train'
train_list=os.listdir(train_dir)
train_list.sort()
train_feature={'img_name':[],'mid_feature':[],'hig_feature':[]}
###########target directory################

with torch.no_grad():
    for i in range(len(train_list)):
        sub_dir=train_dir+'/'+train_list[i]
        sub_list=os.listdir(sub_dir)
        sub_list.sort()
        print('train processing {}'.format(train_list[i]))
        for j in range(len(sub_list)):
            img_path=sub_dir+'/'+sub_list[j]
            img0=Image.open(img_path)
            img1=trans(img0)
            img2=torch.unsqueeze(img1,0)
            img2=img2.cuda()
            outputs=model(img2)
            
            mid_f=pool(forward_outputs[0])
            mid_f=torch.squeeze(mid_f)
            mid_f=mid_f.cpu().data.numpy()
            
            hig_f=forward_outputs[1]
            hig_f=torch.squeeze(hig_f)
            hig_f=hig_f.cpu().data.numpy()            
 
            del forward_outputs[-2:]
            
            train_feature['mid_feature'].append(mid_f)
            train_feature['hig_feature'].append(hig_f)
            train_feature['img_name'].append(sub_list[j])

with open('train_feature.pickle','wb') as f:
    pickle.dump(train_feature,f)
        
        
###########target directory################
val_dir='../data/val'
val_list=os.listdir(val_dir)
val_list.sort()
val_feature={'img_name':[],'mid_feature':[],'hig_feature':[]}
###########target directory################

with torch.no_grad():
    for i in range(len(val_list)):
        sub_dir=val_dir+'/'+val_list[i]
        sub_list=os.listdir(sub_dir)
        sub_list.sort()
        print('val processing {}'.format(val_list[i]))
        for j in range(len(sub_list)):
            img_path=sub_dir+'/'+sub_list[j]
            img0=Image.open(img_path)
            img1=trans(img0)
            img2=torch.unsqueeze(img1,0)
            img2=img2.cuda()
            outputs=model(img2)
            
            mid_f=pool(forward_outputs[0])
            mid_f=torch.squeeze(mid_f)
            mid_f=mid_f.cpu().data.numpy()
            
            hig_f=forward_outputs[1]
            hig_f=torch.squeeze(hig_f)
            hig_f=hig_f.cpu().data.numpy()            
 
            del forward_outputs[-2:]
            
            val_feature['mid_feature'].append(mid_f)
            val_feature['hig_feature'].append(hig_f)
            val_feature['img_name'].append(sub_list[j])

with open('val_feature.pickle','wb') as f:
    pickle.dump(train_feature,f)    
            
            

   
            

