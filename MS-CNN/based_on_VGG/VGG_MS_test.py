#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 08:14:55 2019

@author: d
"""

from __future__ import print_function, division

import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets,models, transforms
import os
from PIL import Image
import pickle

###########multi-net####################
class multi_vgg(nn.Module):
    def __init__(self,model,num_class=45):
        super(multi_vgg,self).__init__()
        self.feature=model.features
        self.pool=model.avgpool
        self.classifier=model.classifier
        
        self.distribution=nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_class),
        )
        
        for m in self.distribution.modules():
            if isinstance(m,nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self,x):
        x=self.feature(x)
        x=self.pool(x)
        x = x.view(x.size(0), -1)
        out1=self.classifier(x)
        out2=self.distribution(x)
  
        return out1,out2
###########multi-net####################
        
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
model1=models.vgg16_bn(pretrained=True)
num_ftrs=model1.classifier[6].in_features
model1.classifier[6]=nn.Linear(num_ftrs,num_classes)
model1=model1.cuda()
model2=multi_vgg(model1,num_classes)
model2 = model2.cuda()

wts=torch.load('./train_MS/w_0.pth')
model2.load_state_dict(wts)
model2.eval()
#############model and load weight######################

###########test dir######################
test_dir='../data/test'
test_list=os.listdir(test_dir)
test_list.sort()
###########test dir######################

###########result record###############
cfmtx=np.zeros((num_classes,num_classes))
recall=np.zeros((num_classes))
precision=np.zeros((num_classes))
acc=0.0
###########result record###############

with torch.no_grad():
    for i in range(len(test_list)):
        sub_dir=test_dir+'/'+test_list[i]
        srcidx=class_names.index(test_list[i])
        sub_list=os.listdir(sub_dir)
        sub_list.sort()
        print('class {} processing'.format(test_list[i]))
        for j in range(len(sub_list)):
            img_path=sub_dir+'/'+sub_list[j]
            img0=Image.open(img_path)
            img1=trans(img0)
            img2=torch.unsqueeze(img1,0)
            img2=img2.cuda()
            outputs=model2(img2)
            out_cls=outputs[0]
            out_cls=out_cls.cpu().data.numpy()[0]
            dstidx=np.argmax(out_cls)
            cfmtx[srcidx,dstidx]+=1
            
acc=np.sum(np.diag(cfmtx))/np.sum(cfmtx)        
for c in range(num_classes):
    counts=cfmtx[c,:].sum()
    recall[c,]=cfmtx[c,c]/counts
    precision[c,]=cfmtx[c,c]/np.sum(cfmtx[:,c])

result={'acc':acc,'recall':recall,'precision':precision,'confuse':cfmtx}
with open('VGG_MS_test.pickle','wb') as f:
    pickle.dump(result,f)

