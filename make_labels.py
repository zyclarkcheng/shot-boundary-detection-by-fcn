# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 17:04:51 2018

@author: clark
"""

'''
Make labels for temporal snippet
a snippet is [depth,64,64,3]
corresponnding label shape is is [1,2], value is [1,0] or [0,1]
means frame 6 is same with frame 5, or not same.
'''
import numpy as np
depth=10
labels808_2=np.load('./labels808_2.npy')
len_files=np.shape(labels808_2)[0]
len_labels=len_files-depth
labels=np.zeros([len_labels,1])
mid=int(np.floor(depth/2)+1)
for i in range(len_labels):
        difference=labels808_2[i+mid,0]+labels808_2[i+mid+1,0]+labels808_2[i+mid+2,0]-  \
        labels808_2[i+mid-1,0]-labels808_2[i+mid-2,0]-labels808_2[i+mid-3,0]
        
        labels[i,0]=1 if (difference>0) else 0
        