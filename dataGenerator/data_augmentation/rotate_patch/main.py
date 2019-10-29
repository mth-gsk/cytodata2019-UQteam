#%%
import numpy as np
from random import choice


def random_rotation(patch):
        'Rotate a patch and its label with a random multiple of 90 degrees'
            
        angle = choice([0, 90, 180, 270])
        # Rotate patch  
        if angle == 0:
            rotated_patch = patch
        elif angle == 90:
            rotated_patch = np.rot90(patch, -1)
        elif angle == 180:
            rotated_patch = np.rot90(patch, -2)
        elif angle == 270:
            rotated_patch = np.rot90(patch, 1)
        else:
            raise ValueError("Unkown Angle should be 0, 90, 180 or 270") 

        return rotated_patch