#%%
from dataGenerator import DataGenerator
from personal_data import location_data

dataGenerator = DataGenerator(location_data,
batch_size=128)

k = dataGenerator.generate()

batch = next(k)