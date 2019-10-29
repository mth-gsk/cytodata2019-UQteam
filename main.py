#%%
from dataGenerator import DataGenerator
from personal_data import location_data
import numpy as np
from models import simple_model

model = simple_model(17, (201, 201))
dataGenerator = DataGenerator(location_data,
inputDim=(100, 100),
batch_size=128)

k = dataGenerator.generate()

batch = next(k)
