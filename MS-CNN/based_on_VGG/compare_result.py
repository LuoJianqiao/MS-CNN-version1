#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 13:39:34 2019

@author: d
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

with open('VGG_test.pickle','rb') as f1:
    result1=pickle.load(f1)

acc1=result1['acc']
recall1=result1['recall']
mr1=np.mean(recall1)
precision1=result1['precision']
mp1=np.mean(precision1)
confuse1=result1['confuse']
dia1=np.diag(confuse1)

with open('VGG_MS_test.pickle','rb') as f2:
    result2=pickle.load(f2)

acc2=result2['acc']
recall2=result2['recall']
mr2=np.mean(recall2)
precision2=result2['precision']
mp2=np.mean(precision2)
confuse2=result2['confuse']
dia2=np.diag(confuse2)

root_dir='../data/train'
class_name=os.listdir(root_dir)
class_name.sort()

class_num=confuse1.shape[0]
for i in range(class_num):
    confuse1[i,:]=confuse1[i,:]/confuse1[i,:].sum()
    confuse2[i,:]=confuse2[i,:]/confuse2[i,:].sum()

     
dot_list1=[]
dot_list2=[]
txt1=[]
txt2=[]
for i in range(class_num):
    for j in range(class_num):
        if confuse1[i,j]>0.01:
            dot_list1.append([j,i,confuse1[i,j]])
            v1=round(confuse1[i,j],3)
            if v1>0.99:
                txt1.append('1')
            else:            
                string1=str(v1)
                string1=string1[1:4]
                txt1.append(string1)
        
        if confuse2[i,j]>0.01:
            dot_list2.append([j,i,confuse2[i,j]])
            v2=round(confuse2[i,j],3)
            if v2>0.99:
                txt2.append('1')
            else:
                string2=str(v2)
                string2=string2[1:4]
                txt2.append(string2)

dot_list1=np.array(dot_list1)
dot_list2=np.array(dot_list2)

fig1 = plt.figure('original CNN')
ax1 = fig1.add_subplot(1, 1, 1)
tick=list(range(class_num))
ax1.set_xticks(tick)
ax1.set_xticklabels(class_name,rotation=90, fontsize=10)
ax1.set_xlabel('predicted')
ax1.set_yticks(tick)
ax1.set_yticklabels(class_name,rotation=0, fontsize=10)
ax1.set_ylabel('true')
ax1.imshow(confuse1,cmap='binary')
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



fig2 = plt.figure('MS-CNN')
ax2 = fig2.add_subplot(1, 1, 1)
ax2.set_xticks(tick)
ax2.set_xticklabels(class_name,rotation=90, fontsize=10)
ax2.set_xlabel('predicted')
ax2.set_yticks(tick)
ax2.set_yticklabels(class_name,rotation=0, fontsize=10)
ax2.set_ylabel('true')
ax2.imshow(confuse2,cmap='binary')


for i in range(len(txt2)):
    if dot_list2[i,2]>0.4:
        ax2.annotate(txt2[i], xy = (dot_list2[i,0], \
                dot_list2[i,1]),\
                xytext = (dot_list2[i,0]-0.5, dot_list2[i,1]+0.25),\
                color='white',fontsize=7)
       
    if dot_list2[i,2]<0.4:
        ax2.annotate(txt2[i], xy = (dot_list2[i,0], \
                    dot_list2[i,1]),\
                    xytext = (dot_list2[i,0]-0.5, dot_list2[i,1]+0.25),\
                    color='black',fontsize=7)

####################process confuse##########################
#
################process diag#######################
fig3=plt.figure('recall')
ax3=fig3.add_subplot(1,1,1)
ax3.set_xticks(tick)
ax3.set_xticklabels(class_name,rotation=90, fontsize=7)
ax3.set_xlabel('class name')
ax3.set_ylabel('acc')
ax3.bar(np.arange(class_num)-0.15,np.diag(confuse1),width=0.3,label='original CNN')
ax3.bar(np.arange(class_num)+0.15,np.diag(confuse2),width=0.3,label='MS-CNN')
ax3.legend()

val_acc1=np.load('./train_hard/acc.npy')
val_acc2=np.load('./train_MS/acc.npy')
fig4=plt.figure('validation_acc')
ax4=fig4.add_subplot(1,1,1)
ax4.plot(range(100),val_acc1,label='original CNN')
ax4.plot(range(100),val_acc2,label='MS-CNN')
ax4.set_xlabel('epoch')
ax4.set_ylabel('acc')
ax4.legend()
