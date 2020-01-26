#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 19:31:52 2019

@author: a
"""
import os
import shutil
import numpy as np


img_dir='path of the NR dataset'
#other classification datasets is also avilable, such as the AID, CIFAR100, VOC and so on.#
training_ratio=0.1
validation_ratio=0.1
testing_ratio=0.8

save_dir='.'

sub_dir_list=os.listdir(img_dir)
sub_dir_list.sort()
for name in sub_dir_list:
    train_dir=save_dir+'/train/'+name
    if not os.path.exists(train_dir):
        os.mkdir(train_dir)
        
    test_dir=save_dir+'/test/'+name
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)
        
    val_dir=save_dir+'/val/'+name
    if not os.path.exists(val_dir):
        os.mkdir(val_dir)

train_len_list=[]
val_len_list=[]
test_len_list=[]
for sub in sub_dir_list:
    print('class {}'.format(sub))
    img_sub=img_dir+'/'+sub
    img_list=os.listdir(img_sub)   
    img_list.sort()
    dir_len=len(img_list)
    
    train_num=int(dir_len*training_ratio)
    val_num=int(dir_len*validation_ratio)
    test_num=int(dir_len*testing_ratio)
    
    train_len_list.append(train_num)
    val_len_list.append(val_num)
    test_len_list.append(test_num)

    
    allidx=list(range(dir_len))
    trainidx=np.random.choice(allidx,size=train_num,replace=False)
    trainlist=trainidx.tolist()
    trainlist.sort()
    
    for i in trainlist:
        allidx.remove(i)        
    validx=np.random.choice(allidx,size=val_num,replace=False)
    vallist=validx.tolist()
    vallist.sort()
       
    for i in vallist:
        allidx.remove(i)
    
    testlist=allidx[:]
    testlist.sort()

       
    for i1 in range(len(trainlist)):
        img_name=img_list[trainlist[i1]]
        img_path=img_dir+'/'+sub+'/'+img_name
        save_path=save_dir+'/train/'+sub+'/'+img_name
        shutil.copy(img_path,save_path)
        
    for i2 in range(len(vallist)):
        img_name=img_list[vallist[i2]]
        img_path=img_dir+'/'+sub+'/'+img_name
        save_path=save_dir+'/val/'+sub+'/'+img_name
        shutil.copy(img_path,save_path)
        
    for i3 in range(len(testlist)):
        img_name=img_list[testlist[i3]]
        img_path=img_dir+'/'+sub+'/'+img_name
        save_path=save_dir+'/test/'+sub+'/'+img_name
        shutil.copy(img_path,save_path)

train_list=np.array(train_len_list)
val_list=np.array(val_len_list)
test_list=np.array(test_len_list)
np.save('train_list.npy',train_list)
np.save('val_list.npy',val_list)
    
shutil.copy('train_list.npy','../based_on_Alex/soft_label/train_list.npy')     
shutil.copy('val_list.npy','../based_on_Alex/soft_label/val_list.npy')   

shutil.copy('train_list.npy','../based_on_ResNet/soft_label/train_list.npy')     
shutil.copy('val_list.npy','../based_on_ResNet/soft_label/val_list.npy') 

shutil.copy('train_list.npy','../based_on_VGG/soft_label/train_list.npy')     
shutil.copy('val_list.npy','../based_on_VGG/soft_label/val_list.npy')   