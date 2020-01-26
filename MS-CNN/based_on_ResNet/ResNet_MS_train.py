#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 17:33:56 2019

@author: d
"""

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy


###########multi-net####################
class multi_resnet(nn.Module):
    def __init__(self,model,num_class=45):
        super(multi_resnet,self).__init__()
        self.feature=nn.Sequential(*list(model.children())[:-1])
        self.classifier=model.fc        
        self.distribution=nn.Linear(2048, num_class)
        
        for m in self.distribution.modules():
            if isinstance(m,nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
        
    def forward(self,x):
        x=self.feature(x)
        x = x.reshape(x.size(0), -1)
        out1=self.classifier(x)
        out2=self.distribution(x)     
        return out1,out2
###########multi-net####################


############class prob#################
class_prob=np.load('./soft_label/soft_label.npy')
class_weight=torch.as_tensor(class_prob,dtype=torch.float32,device='cuda:0')      
############class prob#################

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
model1=models.resnet101(pretrained=True)
num_ftrs=model1.fc.in_features
model1.fc=nn.Linear(num_ftrs,num_classes)
model1 = model1.cuda()
model2=multi_resnet(model1,num_classes)
model2=model2.cuda()
#############model######################

############criteria ,optimizer, learn rate####################
criterion = nn.CrossEntropyLoss()
cls_params=list(map(id,model2.classifier.parameters()))
dis_params=list(map(id,model2.distribution.parameters()))
fc_params=cls_params+dis_params

base_params=filter(lambda p:id(p) not in fc_params,
                   model2.parameters())
opt = optim.SGD([{'params':base_params},
                 {'params':model2.classifier.parameters(),'lr':0.01},
                 {'params':model2.distribution.parameters(),'lr':0.01}]
                , lr=0.001, momentum=0.9,weight_decay=5e-4)
lr_v=lr_scheduler.ReduceLROnPlateau(opt,mode='min',factor=0.1,patience=3,verbose=True)
############criteria ,optimizer, learn rate####################

############para initialise####################
wts_val=copy.deepcopy(model2.state_dict())
num_epochs=100
lamda=1.4#the optimal lamda in different cases is given in the paper at Table 6
############para initialise####################

#########record and result##################    
loss_history=np.zeros((num_epochs,))
acc_history=np.zeros((num_epochs,))
Sn=5# the checkpoint accounting the 5th highest accuracy on the validation is regarded as the final model
result_dir='./train_MS'# path to save the training results, including accuracy curve, loss curve, and weights
if not os.path.exists(result_dir):
    os.mkdir(result_dir)    
#########record and result################## 

for epoch in range(num_epochs):
    
    print('ep0: {}/{}'.format(epoch+1,num_epochs))   
    print('phase: train')
    model1.train()
    model2.train()
    epo_loss=0.0   
    since=time.time()
    ####################train########################
    for inputs,labels in dataloaders['train']:
      
        inputs=inputs.cuda()
        labels=labels.cuda()
        nlabels=labels.cpu().data.numpy()
        
        opt.zero_grad() 
        
        outputs=model2(inputs)
        out_cls=outputs[0]
        out_dis=outputs[1]
        
        loss_cls=criterion(out_cls,labels)        
        prob=nn.functional.softmax(out_dis,dim=1)
        row=out_dis.size(0)
        col=out_dis.size(1)
        loss_dis=torch.tensor(0,dtype=torch.float32,device='cuda:0')       
        for b in range(row):
            truth=int(nlabels[b])
            temp_weight=class_weight[truth]   
            for c in range(col):
              if temp_weight[c,]>0 and prob[b,c]>0.001:
                  loss_dis-=torch.mul(torch.log(prob[b,c]),temp_weight[c,])        
        loss_dis/=row
        loss=loss_cls+lamda*loss_dis
        
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
    model1.eval()
    model2.eval()
    running_corrects = 0.0
    for inputs,labels in dataloaders['val']:
        inputs=inputs.cuda()
        labels=labels.cuda()
        outputs=model2(inputs)
        out_cls=outputs[0]
        _, preds = torch.max(out_cls, 1)
        running_corrects += torch.sum(preds == labels.data)
    
    epoch_acc = running_corrects.double() / dataset_sizes['val']
    acc=epoch_acc.cpu().data.numpy()
    acc_history[epoch,]=acc

    if epoch<Sn:
        wts_path=result_dir+'/w_'+str(epoch)+'.pth'
        wts_val=copy.deepcopy(model2.state_dict())
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
            wts_val=copy.deepcopy(model2.state_dict())
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