import random
import numpy as np
from math import ceil, floor

def random_crop(img, newDim, maxOffset):
    
    if maxOffset > 0:
        offsetX = random.randrange(-maxOffset, maxOffset)
        offsetY = random.randrange(-maxOffset, maxOffset)
    else:
        offsetX = 0
        offsetY = 0
    
    x1 = - (newDim[0] - img.shape[0])/2 + offsetX
    y1 = - (newDim[1] - img.shape[1])/2 + offsetY
    x2 = x1 + newDim[0]
    y2 = y1 + newDim[1]

    # Round the number to fit exactly with the pixels
    x1 = int(ceil(x1))
    y1 = int(ceil(y1))
    x2 = int(ceil(x2))
    y2 = int(ceil(y2))

    random_crop = img[x1:x2, y1:y2]

    return random_crop