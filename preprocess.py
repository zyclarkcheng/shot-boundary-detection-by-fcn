# -*- coding: utf-8 -*-
"""
Created on Sun Jun  3 21:18:15 2018

@author: clark
"""

'''
preprocess data
1 read from files
2 use mask, resize,
3 make corresponding label 
4 concatenate single image into 11 frames
4 save as npy
'''
import cv2
import os
import glob
import numpy as np

def process_single(file,img_size):
    mask=cv2.imread('/home/clark/umich_3/umichbiological_DB_1000_02.tif',0)
    mask=np.abs(mask-255)
    image=cv2.imread(file)
    image= cv2.bitwise_and(image,image,mask = mask)
    #RGB_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #plt.imshow(image)
    image = cv2.resize(image, (img_size, img_size),0,0, cv2.INTER_LINEAR)
    image = image.astype(np.float32)
    image = np.multiply(image, 1.0 / 255.0)
    return image
    
#folders=['01','02','03','04','05','06','07','08','09','10','11','12']
data_path='/home/clark/umich_3/2010/data/'
image_size = 64
len_snippets=10
images = []
labels = []    
path=os.path.join(data_path,'*g')
files=sorted(glob.glob(path))
len_files=len(files)

for file in files:
        image=process_single(file,image_size)
        images.append(image)

images=np.array(images)
labels=np.zeros(len_files,1)
#np.save('images',images)
#imagesArr=np.concatenate((imagesArr,images))
#    