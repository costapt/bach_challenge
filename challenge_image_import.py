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
#iii_benign = np.zeros([tam_class*3, 224, 224, 3])
#iii_insitu = np.zeros([tam_class*3, 224, 224, 3])
#iii_invasive = np.zeros([tam_class*3, 224, 224, 3])
#iii_normal = np.zeros([tam_class*3, 224, 224, 3])
#
l_benign = np.zeros([tam_class*3, 1])
l_insitu = np.zeros([tam_class*3, 1])
l_invasive = np.zeros([tam_class*3, 1])
l_normal = np.ones([tam_class*3,1])
#
#for i in range(tam_class):
#    iii_benign[i,:,:,:] = ii_benign[i, :, 0:224, :]
#    iii_benign[i+200,:,:,:] = ii_benign[i, :, 299-224:299, :]
#    iii_benign[i+100,:,:,:] = ii_benign[i, :, 150-112:150+112, :]
#
#    iii_insitu[i,:,:,:] = ii_insitu[i, :, 0:224, :]
#    iii_insitu[i+200,:,:,:] = ii_insitu[i, :, 299-224:299, :]
#    iii_insitu[i+100,:,:,:] = ii_insitu[i, :, 150-112:150+112, :]
#
#    iii_invasive[i,:,:,:] = ii_invasive[i, :, 0:224, :]
#    iii_invasive[i+200,:,:,:] = ii_invasive[i, :, 299-224:299, :]
#    iii_invasive[i+100,:,:,:] = ii_invasive[i, :, 150-112:150+112, :]
#
#    iii_normal[i,:,:,:] = ii_normal[i, :, 0:224, :]
#    iii_normal[i+200,:,:,:] = ii_normal[i, :, 299-224:299, :]
#    iii_normal[i+100,:,:,:] = ii_normal[i, :, 150-112:150+112, :]

with open('i_benign.pickle', 'rb') as handle:
    iii_benign = pickle.load(handle)
with open('i_insitu.pickle', 'rb') as handle:
    iii_insitu = pickle.load(handle)
with open('i_invasive.pickle', 'rb') as handle:
    iii_invasive = pickle.load(handle)
with open('i_normal.pickle', 'rb') as handle:
    iii_normal = pickle.load(handle)

from sklearn.model_selection import train_test_split
benign_i_train, benign_i_test, benign_l_train, benign_l_test = train_test_split(iii_benign, l_benign, test_size = 0.05, random_state = 20)
insitu_i_train, insitu_i_test, insitu_l_train, insitu_l_test = train_test_split(iii_insitu, l_insitu, test_size = 0.05, random_state = 20)
invasive_i_train, invasive_i_test, invasive_l_train, invasive_l_test = train_test_split(iii_invasive, l_invasive, test_size = 0.05, random_state = 20)
normal_i_train, normal_i_test, normal_l_train, normal_l_test = train_test_split(iii_normal, l_normal, test_size = 0.15, random_state = 20)
#
other_i_train = np.concatenate((benign_i_train, insitu_i_train, invasive_i_train),0)
other_i_test = np.concatenate((benign_i_test, insitu_i_test, invasive_i_test),0)
other_l_train = np.concatenate((benign_l_train, insitu_l_train, invasive_l_train),0)
other_l_test = np.concatenate((benign_l_test, insitu_l_test, invasive_l_test),0)

oo_i_train, oo_i_delete, oo_l_train, oo_l_delete = train_test_split(other_i_train, other_l_train, test_size = 1 - (255/855), random_state = 20)

#i_train = np.concatenate((normal_i_train, other_i_train),0)
i_train = np.concatenate((normal_i_train, oo_i_train),0)
i_test = np.concatenate((normal_i_test, other_i_test),0)
#l_train = np.concatenate((normal_l_train, other_l_train),0)
l_train = np.concatenate((normal_l_train, oo_l_train),0)
l_test = np.concatenate((normal_l_test, other_l_test),0)

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

model = Sequential()

model.add(Conv2D(128, (5,5), activation='relu', input_shape=(224,224,3)))
model.add(Conv2D(128, (5,5), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(256, (5,5), activation='relu'))
model.add(Conv2D(256, (5,5), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(512, (5,5), activation='relu'))
model.add(Conv2D(512, (5,5), activation='relu'))
#model.add(Conv2D(256, (3,3), activation='relu', padding='same'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(1024, (5,5), activation='relu'))
model.add(Conv2D(1024, (5,5), activation='relu'))
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
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# 8. Compile model
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

#class_weight = {0 : 0.65, 1 : 2.18}
model.fit(i_train, l_train, batch_size = 8, validation_data = (i_test, l_test), epochs=200, verbose = 1, shuffle = True)#, class_weight = class_weight)
