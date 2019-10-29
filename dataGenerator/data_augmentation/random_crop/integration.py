#%%
from dataGenerator.data_augmentation.random_crop import random_crop
import numpy as np

inputImg = np.zeros((25, 25), dtype=np.uint8)
inputImg[10, 15] = 255

smallImg = random_crop(inputImg, (15, 15), 0)