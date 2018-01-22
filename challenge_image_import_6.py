#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 11:19:18 2017

@author: cferreira
"""

import numpy as np
import pickle

tam_val = 10
tam_train = 70
tam_test = 20

with open('benign_square_validate.pickle', 'rb') as handle:
    benign_i_val = pickle.load(handle)
with open('benign_square_train.pickle', 'rb') as handle:
    benign_i_train = pickle.load(handle)
with open('benign_square_test.pickle', 'rb') as handle:
    benign_i_test = pickle.load(handle)
with open('insitu_square_validate.pickle', 'rb') as handle:
    insitu_i_val = pickle.load(handle)
with open('insitu_square_train.pickle', 'rb') as handle:
    insitu_i_train = pickle.load(handle)
with open('insitu_square_test.pickle', 'rb') as handle:
    insitu_i_test = pickle.load(handle)
with open('invasive_square_validate.pickle', 'rb') as handle:
    invasive_i_val = pickle.load(handle)
with open('invasive_square_train.pickle', 'rb') as handle:
    invasive_i_train = pickle.load(handle)
with open('invasive_square_test.pickle', 'rb') as handle:
    invasive_i_test = pickle.load(handle)
with open('normal_square_validate.pickle', 'rb') as handle:
    normal_i_val = pickle.load(handle)
with open('normal_square_train.pickle', 'rb') as handle:
    normal_i_train = pickle.load(handle)
with open('normal_square_test.pickle', 'rb') as handle:
    normal_i_test = pickle.load(handle)

i_train = np.concatenate((benign_i_train, insitu_i_train, invasive_i_train, normal_i_train),0)
i_val = np.concatenate((benign_i_val, insitu_i_val, invasive_i_val, normal_i_val),0)
i_test = np.concatenate((benign_i_test, insitu_i_test, invasive_i_test, normal_i_test),0)

ll_train = np.zeros([len(i_train),4])
ll_val = np.zeros([len(i_val),4])
ll_test = np.zeros([len(i_test),4])

for i in range(len(i_train)):
    if (i <  tam_train):
        ll_train[i,:] = [1, 0, 0, 0]
    elif (i>= tam_train) & (i < 2*tam_train):
        ll_train[i,:] = [0, 1, 0, 0]
    elif (i>= 2*tam_train) & (i < 3*tam_train):
        ll_train[i,:] = [0, 0, 1, 0]
    elif (i >= tam_train*3):
        ll_train[i,:] = [0, 0, 0, 1]

for i in range(len(i_val)):
    if (i <  tam_val):
        ll_val[i,:] = [1, 0, 0, 0]
    elif (i>= tam_val) & (i < 2*tam_val):
        ll_val[i,:] = [0, 1, 0, 0]
    elif (i>= 2*tam_val) & (i < 3*tam_val):
        ll_val[i,:] = [0, 0, 1, 0]
    elif (i >= tam_val*3):
        ll_val[i,:] = [0, 0, 0, 1]

for i in range(len(i_test)):
    if (i <  tam_test):
        ll_test[i,:] = [1, 0, 0, 0]
    elif (i>= tam_test) & (i < 2*tam_test):
        ll_test[i,:] = [0, 1, 0, 0]
    elif (i>= 2*tam_test) & (i < 3*tam_test):
        ll_test[i,:] = [0, 0, 1, 0]
    elif (i >= tam_test*3):
        ll_test[i,:] = [0, 0, 0, 1]

from keras.applications.inception_resnet_v2 import InceptionResNetV2#, preprocess_input
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.models import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping
import os

import gc
gc.collect()

base_model = InceptionResNetV2(include_top = 0, weights='imagenet', input_shape=[224,224,3], classes = 4)
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(256, activation = 'relu')(x)
x = Dropout(0.5)(x)
x = Dense(4, activation = 'softmax')(x)
finetuned_model = Model(base_model.input, x)

for layer in base_model.layers:
    layer.trainable = False

top_weights_path = os.path.join('/home/cferreira/python/challenge', 'top_model_weights.h5')
final_weights_path = os.path.join('/home/cferreira/python/challenge', 'model_weights.h5')

callbacks_list = [
        ModelCheckpoint(top_weights_path, monitor='val_loss', verbose=1, save_best_only=True),
        EarlyStopping(monitor='val_loss', patience=20, verbose=0)
    ]

callbacks_list_2 = [
        ModelCheckpoint(final_weights_path, monitor='val_loss', verbose=1, save_best_only=True),
#        EarlyStopping(monitor='val_loss', patience=20, verbose=0)
    ]

from keras.optimizers import SGD
finetuned_model.compile(optimizer = SGD(lr = 0.02), loss = 'categorical_crossentropy', metrics=['accuracy'])

from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
        featurewise_center=False,
        featurewise_std_normalization=False,
        rotation_range=18,
        horizontal_flip = True,
        vertical_flip = True,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1)#,
#        preprocessing_function=preprocess_input)

hist = finetuned_model.fit_generator(datagen.flow(i_train, ll_train, batch_size = 16, shuffle = True), validation_data = (i_val, ll_val), steps_per_epoch = len(i_train), epochs = 75, callbacks = callbacks_list, verbose = 1)

finetuned_model.load_weights(top_weights_path)

for layer in finetuned_model.layers[:679]:
    layer.trainable = False
for layer in finetuned_model.layers[679:]:
    layer.trainable = True

from keras.optimizers import SGD
finetuned_model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

hist = finetuned_model.fit_generator(datagen.flow(i_train, ll_train, batch_size = 16, shuffle = True), validation_data = (i_val, ll_val), steps_per_epoch = len(i_train), epochs = 75, callbacks = callbacks_list_2, verbose = 1)
