import os

import cv2
import keras
import numpy as np

from .data_augmentation import random_crop, random_rotation


class TestDataGenerator(keras.utils.Sequence):
    def __init__(
        self,
        locationImages: str,
        batch_size=1,
        inputDim=(201, 201),
        n_channels=2,
        patch_per_img=1,
        cropOffset=0,
        normalize_data=0,
    ):

        # storing imported parameters
        self.inputDim = inputDim
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.patch_per_img = patch_per_img
        self.cropOffset = cropOffset
        self.normalize_data = normalize_data

        self.allData = [os.path.join(locationImages, filename)
                        for filename in os.listdir(locationImages)]

    def __len__(self):
        "Denotes the number of batches"
        return np.int(np.ceil(len(self.allData) / self.batch_size))

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

    def get_patch(self, imgLocation):
        "Gets one patch of image with given ID"
        # Initialization
        X = np.empty((1, *self.inputDim, self.n_channels))

        # Get input
        img = cv2.imread(imgLocation)[:, :, :self.n_channels]

        img = random_crop(img, self.inputDim, self.cropOffset)

        if self.normalize_data == 1:
            X[0] = img / np.max(img)
        elif self.normalize_data == 2:
            # https://github.com/pytorch/examples/blob/97304e232807082c2e7b54c597615dc0ad8f6173/imagenet/main.py#L197-L198
            X[0] = (img / 255 - [0.485, 0.456, 0.406]) - [0.229, 0.224, 0.225]
        else:
            X[0] = img / 255
        return X

    def __get_batch(self, index):
        X = np.empty((self.batch_size, *self.inputDim, self.n_channels))

        i = index * self.batch_size
        this_batch_size = np.minimum(self.batch_size, len(self.allData) - i)
        for b in range(this_batch_size):
            filename = self.allData[i + b]
            X[b] = self.get_patch(filename)

        return X
