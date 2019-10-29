#%%
from dataGenerator import DataGenerator

dataGenerator = DataGenerator(r"C:\Users\tomni\Pictures\hackaton2019\training",
batch_size=128)

k = dataGenerator.generate()

batch = next(k)