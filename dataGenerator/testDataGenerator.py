import os

import cv2
import keras
import numpy as np


class TestDataGenerator(keras.utils.Sequence):
    def __init__(
        self,
        locationImages: str,
        batch_size=1,
        inputDim=(201, 201),
        n_channels=2,
        patch_per_img=1,
        cropOffset=0,
    ):

        # storing imported parameters
        self.inputDim = inputDim
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.patch_per_img = patch_per_img
        self.cropOffset = cropOffset

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
        img = cv2.imread(imgLocation)[:, :, :2]

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
