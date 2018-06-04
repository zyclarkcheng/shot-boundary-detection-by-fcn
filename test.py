# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 16:10:46 2018

@author: clark
"""
import tensorflow as tf
import numpy as np

image_size=64
depth=10
learning_rate=0.0001
train_from_scratch=1
training_epochs=50
image_path='./images808_64_64_3.npy'
label_path='./labels808_2.npy'
images=np.load(image_path)
labels=np.load(label_path)
len_files=np.shape(labels)[0]
batch_size=4

#for epoch in range(training_epochs):
avg_cost = 0.
total_snippets = len_files-depth
total_batch=int(total_snippets/batch_size)
# Loop over all batches
#            total_pred=np.zeros([0,2])
for s in range(12):#range(total_snippets-1):
    batch_x=np.zeros([0,depth,image_size,image_size,3])
    for i in range(4):
        batch_x=np.concatenate((batch_x,np.expand_dims(images[s:s+depth,:,:,:],axis=0)))
        
#                batch_x = images[batch_size,i]
#                batch_y = labels[i*batch_size:(i+1)*batch_size]