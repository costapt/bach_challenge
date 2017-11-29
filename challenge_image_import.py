#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 11:19:18 2017

@author: cferreira
"""

#folder = '/home/cferreira/Desktop/challenge/Benign/'
#
#from os import listdir
#from PIL import Image as PImage
import numpy as np
import pickle

tam_class = 100
#
#def loadImages(path):
#    # return array of images
#
#    i = 0
#    i_case = np.zeros([tam_class, 1536, 2048, 3])
#    imagesList = listdir(path)
#    imagesList.sort()
#    #loadedImages = []
#    for image in imagesList:
#        if (image.endswith(".tif")):
##            imagesList.remove(image)
##        else:
#            img = PImage.open(path + image)
##            print(path+image)
#            img2 = np.array(img)
#            img2 = img2/255
#            i_case[i,:,:,:] = img2
#            i += 1
#            #loadedImages.append(img2)
#    
#    return i_case#loadedImages
#
## your images in an array
#f_benign = '/home/cferreira/Desktop/challenge/Benign/'
#f_insitu = '/home/cferreira/Desktop/challenge/InSitu/'
#f_invasive = '/home/cferreira/Desktop/challenge/Invasive/'
#f_normal = '/home/cferreira/Desktop/challenge/Normal/'
#
#i_benign = loadImages(f_benign)
#i_insitu = loadImages(f_insitu)
#i_invasive = loadImages(f_invasive)
#i_normal = loadImages(f_normal)
#
#from scipy.ndimage import zoom
#ii_benign = zoom(i_benign, (1, 299/2048, 299/2048, 1), order = 2)
#ii_insitu = zoom(i_insitu, (1, 299/2048, 299/2048, 1), order = 2)
#ii_invasive = zoom(i_invasive, (1, 299/2048, 299/2048, 1), order = 2)
#ii_normal = zoom(i_normal, (1, 299/2048, 299/2048, 1), order = 2)
#
#iii_benign = np.zeros([tam_class, 3, 224, 224, 3])
#iii_insitu = np.zeros([tam_class, 3, 224, 224, 3])
#iii_invasive = np.zeros([tam_class, 3, 224, 224, 3])
#iii_normal = np.zeros([tam_class, 3, 224, 224, 3])
#
l_benign = np.zeros([tam_class,3,1])
l_insitu = np.zeros([tam_class,3,1])
l_invasive = np.ones([tam_class,3,1])
l_normal = np.zeros([tam_class,3,1])
#
#for i in range(tam_class):
#    iii_benign[i,0,:,:,:] = ii_benign[i, :, 0:224, :]
#    iii_benign[i,2,:,:,:] = ii_benign[i, :, 299-224:299, :]
#    iii_benign[i,1,:,:,:] = ii_benign[i, :, 150-112:150+112, :]
#
#    iii_insitu[i,0,:,:,:] = ii_insitu[i, :, 0:224, :]
#    iii_insitu[i,2,:,:,:] = ii_insitu[i, :, 299-224:299, :]
#    iii_insitu[i,1,:,:,:] = ii_insitu[i, :, 150-112:150+112, :]
#
#    iii_invasive[i,0,:,:,:] = ii_invasive[i, :, 0:224, :]
#    iii_invasive[i,2,,:,:,:] = ii_invasive[i, :, 299-224:299, :]
#    iii_invasive[i,1,,:,:,:] = ii_invasive[i, :, 150-112:150+112, :]
#
#    iii_normal[i,0,:,:,:] = ii_normal[i, :, 0:224, :]
#    iii_normal[i,2,:,:,:] = ii_normal[i, :, 299-224:299, :]
#    iii_normal[i,1,:,:,:] = ii_normal[i, :, 150-112:150+112, :]

with open('i_benign.pickle', 'rb') as handle:
    iii_benign = pickle.load(handle)
with open('i_insitu.pickle', 'rb') as handle:
    iii_insitu = pickle.load(handle)
with open('i_invasive.pickle', 'rb') as handle:
    iii_invasive = pickle.load(handle)
with open('i_normal.pickle', 'rb') as handle:
    iii_normal = pickle.load(handle)

#iii_benign = iii_benign[100:200,:,:,:]
#iii_insitu = iii_insitu[100:200,:,:,:]
#iii_invasive = iii_invasive[100:200,:,:,:]
#iii_normal = iii_normal[100:200,:,:,:]

from sklearn.model_selection import train_test_split
benign_i_train, benign_i_val_test, benign_l_train, benign_l_val_test = train_test_split(iii_benign, l_benign, test_size = 0.12, random_state = 20)
insitu_i_train, insitu_i_val_test, insitu_l_train, insitu_l_val_test = train_test_split(iii_insitu, l_insitu, test_size = 0.12, random_state = 20)
invasive_i_train, invasive_i_val_test, invasive_l_train, invasive_l_val_test = train_test_split(iii_invasive, l_invasive, test_size = 0.18, random_state = 20)
normal_i_train, normal_i_val_test, normal_l_train, normal_l_val_test = train_test_split(iii_normal, l_normal, test_size = 0.12, random_state = 20)

benign_i_val, benign_i_test, benign_l_val, benign_l_test = train_test_split(benign_i_val_test, benign_l_val_test, test_size = 0.75, random_state = 20)
insitu_i_val, insitu_i_test, insitu_l_val, insitu_l_test = train_test_split(insitu_i_val_test, benign_l_val_test, test_size = 0.75, random_state = 20)
invasive_i_val, invasive_i_test, invasive_l_val, invasive_l_test = train_test_split(invasive_i_val_test, invasive_l_val_test, test_size = 0.5, random_state = 20)
normal_i_val, normal_i_test, normal_l_val, normal_l_test = train_test_split(normal_i_val_test, normal_l_val_test, test_size = 0.75, random_state = 20)

other_i_train = np.concatenate((benign_i_train, insitu_i_train, normal_i_train),0)
other_i_test = np.concatenate((benign_i_test, insitu_i_test, normal_i_test),0)
other_l_train = np.concatenate((benign_l_train, insitu_l_train, normal_l_train),0)
other_l_test = np.concatenate((benign_l_test, insitu_l_test, normal_l_test),0)
other_i_val = np.concatenate((benign_i_val, insitu_i_val, normal_i_val),0)
other_l_val = np.concatenate((benign_l_val, insitu_l_val, normal_l_val),0)

i_train = np.concatenate((other_i_train, invasive_i_train),0)
i_test = np.concatenate((other_i_test, invasive_i_test),0)
l_train = np.concatenate((other_l_train, invasive_l_train),0)
l_test = np.concatenate((other_l_test, invasive_l_test),0)
i_val = np.concatenate((other_i_val, invasive_i_val),0)
l_val = np.concatenate((other_l_val, invasive_l_val),0)

ii_train = np.zeros([len(i_train)*3,224,224,3])
ll_train = np.zeros([len(l_train)*3,1])
ii_val = np.zeros([len(i_val)*3,224,224,3])
ll_val = np.zeros([len(l_val)*3,1])


for i in range(len(i_train)):
    ii_train[i,:,:,:] = i_train[i,0,:,:,:]
    ii_train[len(i_train)+i,:,:,:] = i_train[i,1,:,:,:]
    ii_train[len(i_train)*2+i,:,:,:] = i_train[i,2,:,:,:]

    ll_train[i,:] = l_train[i,0,:]
    ll_train[len(i_train)+i,:] = l_train[i,1,:]
    ll_train[len(i_train)*2+i,:] = l_train[i,2,:]

for i in range(len(i_val)):
    ii_val[i,:,:,:] = i_val[i,0,:,:,:]
    ii_val[len(i_val)+i,:,:,:] = i_val[i,1,:]
    ii_val[len(i_val)*2+i,:,:,:] = i_val[i,2,:,:,:]

    ll_val[i,:] = l_val[i,0,:]
    ll_val[len(i_val)+i,:] = l_val[i,1,:]
    ll_val[len(i_val)*2+i,:] = l_val[i,2,:]

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

model = Sequential()
filter_num = 32

model.add(Conv2D(filter_num, (5,5), activation='relu', input_shape=(224,224,3)))
#model.add(Conv2D(filter_num, (5,5), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(filter_num*2, (5,5), activation='relu'))
#model.add(Conv2D(filter_num*2, (5,5), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(filter_num*3, (5,5), activation='relu'))
#model.add(Conv2D(filter_num*3, (5,5), activation='relu'))
#model.add(Conv2D(256, (3,3), activation='relu', padding='same'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(filter_num*4, (5,5), activation='relu'))
#model.add(Conv2D(filter_num*4, (5,5), activation='relu'))
##model.add(Conv2D(512, (3,3), activation='relu', padding='same'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
model.add(Dropout(0.25))
#
#model.add(Conv2D(1024, (3,3), activation='relu'))
#model.add(Conv2D(1024, (3,3), activation='relu'))
##model.add(Conv2D(512, (3,3), activation='relu', padding='same'))
#model.add(MaxPooling2D((2,2), strides=(2,2)))
#model.add(Dropout(0.25))
#
model.add(Flatten())
#model.add(Dense(2048, activation='relu'))
#model.add(Dense(4096, activation='relu'))
model.add(Dense(filter_num*3, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# 8. Compile model
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

class_weight = {0 : 0.66, 1 : 2.11}


history = model.fit(ii_train, ll_train, batch_size = 128, validation_data = (ii_val, ll_val), epochs=600, verbose = 1, shuffle = True, class_weight = class_weight)
    