#%%
from dataGenerator import DataGenerator
from personal_data import location_data

dataGenerator = DataGenerator(location_data,
inputDim=(100, 100),
batch_size=128)

k = dataGenerator.generate()

batch = next(k)