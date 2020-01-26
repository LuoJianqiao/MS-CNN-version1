#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 10:24:47 2019

@author: d
"""

import sys
import os.path
import numpy as np
import scipy.special as ss
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))
import dirichlet.dirichlet as dirichlet
import math
import os
import matplotlib.pyplot as plt

theta=np.load('sample_feature.npy')
for m in range(theta.shape[0]):
    theta[m,:]=theta[m,:]/np.sum(theta[m,:]);

img_dir='the path of the training set'#the path is located at direct "data", subdirect "train"
class_num=len(os.listdir(img_dir)) 
dim=theta.shape[1]
    
############computing alpha################# 
# alpha in this code corresponds the pi in the paper

alpha_cls=np.zeros((class_num,dim))
train_num=np.load('train_list.npy')
val_num=np.load('val_list.npy')
idx=0;
for i in range(class_num):
    num=train_num[i]+val_num[i]
    start=idx
    end=idx+num
    D0=theta[start:end,:]
    alpha=dirichlet.mle(D0)
    alpha_cls[i,:]=alpha
    idx+=num
############computing alpha################# 
    
############computing cls_lnga################# 
cls_lnga=np.zeros(class_num)
for c in range(class_num):
    temp_alpha=alpha_cls[c,:]
    alpha_sum=np.sum(temp_alpha)
    lnga0=math.lgamma(alpha_sum)
    lnga=0
    for k in range(dim):
        tlnga=math.lgamma(temp_alpha[k])
        lnga+=tlnga
    cls_lnga[c,]=lnga0-lnga
    
############computing cls_lnga################# 
 
############computing soft labels################# 
cls_label0=np.zeros((class_num,class_num))
for i in range(class_num):
    alpha_i=alpha_cls[i,:]
    alpha_i0=np.sum(alpha_i)
    for j in range(class_num):        
        alpha_j=alpha_cls[j,:]
        alpha_j0=np.sum(alpha_j)
        tlnga=cls_lnga[j,]
        for k in range(dim):
            tlnga+=(alpha_j[k,]-1)*(ss.psi(alpha_i[k,])-ss.psi(alpha_i0))
        cls_label0[i,j]=np.exp(tlnga)
    cls_label0[i,:]/=np.sum(cls_label0[i,:])
        
cls_label1=cls_label0.copy()
cls_label1[np.where(cls_label0<0.05)]=0
for n in range(cls_label1.shape[0]):
    cls_label1[n,:]=cls_label1[n,:]/np.sum(cls_label1[n,:])

np.save('soft_label.npy',cls_label1)


###########plot the soft label####################
class_name=os.listdir(img_dir)
class_name.sort()
dot_list1=[]
txt1=[]
for i in range(class_num):
    for j in range(class_num):
        if cls_label1[i,j]>0.05:
            dot_list1.append([j,i,cls_label1[i,j]])
            v1=cls_label1[i,j]
            if v1>0.99:
                txt1.append('1')
            else:            
                v2=round(v1,2)
                s1=str(v2)[1:4]
                txt1.append(s1)
        
dot_list1=np.array(dot_list1)

fig1 = plt.figure('soft_label')
ax1 = fig1.add_subplot(1, 1, 1)
tick=list(range(class_num))
ax1.set_xticks(tick)
ax1.set_xticklabels(class_name,rotation=90, fontsize=10)
ax1.set_yticks(tick)
ax1.set_yticklabels(class_name,rotation=0, fontsize=10)
ax1.imshow(cls_label1,cmap='binary')
#ax1.grid()
for i in range(len(txt1)):
    if dot_list1[i,2]>0.4:
        ax1.annotate(txt1[i], xy = (dot_list1[i,0], \
                dot_list1[i,1]),\
                xytext = (dot_list1[i,0]-0.5, dot_list1[i,1]+0.25),\
                color='white',fontsize=7)
    
    
    if dot_list1[i,2]<0.4:
        ax1.annotate(txt1[i], xy = (dot_list1[i,0], \
                    dot_list1[i,1]),\
                    xytext = (dot_list1[i,0]-0.5, dot_list1[i,1]+0.25),\
                    color='black',fontsize=7)
ax1.set_xlabel('semantic distribution')
ax1.set_ylabel('classes')
