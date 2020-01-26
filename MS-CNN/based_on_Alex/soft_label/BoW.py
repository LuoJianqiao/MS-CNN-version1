#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 17:09:48 2019

@author: d
"""

import pickle
import numpy as np
from sklearn.cluster import MiniBatchKMeans as bkmeans
import os
    
with open('../train_feature.pickle','rb') as f1:
    data1=pickle.load(f1,encoding="latin1")
    
with open('../val_feature.pickle','rb') as f2:
    data2=pickle.load(f2,encoding="latin1")

mid_f=data1['mid_feature']+data2['mid_feature']
hig_f=data1['hig_feature']+data2['hig_feature']

mid_set=[]
hig_set=[]

class_num=len(os.listdir('the path of the training set'))#the path is located at direct "data", subdirect "train"
mid_dim=mid_f[0].shape[0]
hig_dim=hig_f[0].shape[0]
word_num=mid_f[0].shape[1]*mid_f[0].shape[2]
for i in range(len(mid_f)):
   
    temp_mid=np.reshape(mid_f[i],(mid_dim,word_num))
    temp_hig=np.reshape(hig_f[i],(hig_dim,word_num))    
    for j in range(word_num):
        mid_set.append(temp_mid[:,j]) 
        hig_set.append(temp_hig[:,j]) 
    if i%100==0:
        print('{} is finished'.format(i))
        
mid_set=np.array(mid_set)
hig_set=np.array(hig_set)

mid_km=bkmeans(n_clusters=800,batch_size=30000,max_iter=10000).fit(mid_set)
hig_km=bkmeans(n_clusters=800,batch_size=30000,max_iter=10000).fit(hig_set)

mid_center=mid_km.cluster_centers_
hig_center=hig_km.cluster_centers_

mid_label=mid_km.labels_
mid_label=np.reshape(mid_label,(-1,word_num))
hig_label=hig_km.labels_
hig_label=np.reshape(hig_label,(-1,word_num))
#############################################################
img_num=len(mid_f)

train_num=np.load('train_list.npy')
idx_train=0
val_num=np.load('val_list.npy')
idx_val=np.sum(train_num)
row=0;

mid_corups=np.zeros((img_num,word_num),dtype=np.int)
hig_corups=np.zeros((img_num,word_num),dtype=np.int)
for cls_idx in range(class_num):
    for img_idx1 in range(train_num[cls_idx]):
            mid_corups[row,:]=mid_label[idx_train]
            hig_corups[row,:]=hig_label[idx_train]
            idx_train+=1
            row+=1

    for img_idx2 in range(val_num[cls_idx]):
            mid_corups[row,:]=mid_label[idx_val]
            hig_corups[row,:]=hig_label[idx_val]
            idx_val+=1
            row+=1


np.save('mid_corups.npy',mid_corups)
np.save('hig_corups.npy',hig_corups)