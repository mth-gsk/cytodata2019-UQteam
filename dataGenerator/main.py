# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 13:57:43 2019

@author: Tom Nijhof
"""
#%%
import cv2
import numpy as np
import keras
import os
import json
import urllib.request
from keras import backend as K
import random
import ast
import math

from skimage.measure import label
from .data_augmentation import random_crop

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, 
                 locationImages: str,
                 batch_size=1, 
                 inputDim=(201,201),
                 n_channels = 2,
                 patch_per_img=1,
                 cropOffset=0,
                 isTrainingData: bool = True,
                 random_rotation=False,
                 flip=False,
                 shuffle=True,
                 border_class=False,
                 illumination_correction=False,
                 ):

        # storing imported parameters
        self.inputDim = inputDim
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.patch_per_img = patch_per_img
        self.cropOffset = cropOffset
        self.random_rotation = random_rotation
        self.flip = flip
        self.shuffle = shuffle
        self.illumination_correction = illumination_correction

        # Setting up connection with LabelBox
        self.labels = [] # will be set later
        self.list_IDs = [] # will be set later

        # needed function
        self.random = random

        # Setting up tmp folder for offline image storage
        self.labelsNames = os.listdir(locationImages)
        
        self.n_labels = len(self.labelsNames)
        self.allData = []

        self.dataPerLabel = []
        for i, labelName in enumerate(self.labelsNames):

            location = os.path.join(locationImages, labelName)
            if isTrainingData:
                allNames = os.listdir(location)[:-100]
            else:
                allNames = os.listdir(location)[-100:]
            self.dataPerLabel.append(
                [
                    os.path.join(location, image) for image in allNames
                ]
            )

        self.on_epoch_end()

    def __len__(self): 
        'Denotes the number of batches per epoch'
        allLenghts = [len(i) for i in self.dataPerLabel]
        return int(np.floor(((np.min(allLenghts)*self.n_labels*self.patch_per_img) / self.batch_size)))
        
    def generate(self):
        """             
        call for yield generator:
        generator_name.generate() INSTEAD of generator_name
        """
        batchNumber = 0
        while True:   
            # Generate data
            yield self.__get_batch(batchNumber)
            batchNumber += 1
            batchNumber %= self.__len__()

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle == True:
            np.random.shuffle(self.allData)
    
        
    def get_patch(self, imgLocation):
        'Gets one patch of image with given ID'
        # Initialization
        X = np.empty((1, *self.inputDim, self.n_channels))

        # Get input
        img = cv2.imread(imgLocation)[:, :, :2]
        
        img = random_crop(img, self.inputDim, self.cropOffset)

        X[0] = img/255
        return X

    def __get_batch(self, index):
        X = np.empty((self.batch_size, *self.inputDim, self.n_channels))
        y = np.empty((self.batch_size, self.n_labels), dtype=int)

        for b in range(self.batch_size):
            randomLabel = random.randint(0, self.n_labels-1)
            numberLabels = len(self.dataPerLabel[randomLabel])
            label = random.randint(0, numberLabels-1)
            X[b] = self.get_patch(self.dataPerLabel[randomLabel][label])
            y[b] = keras.utils.to_categorical(randomLabel, self.n_labels)

        return X, y

