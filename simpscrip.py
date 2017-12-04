import data_processing as dp 
import numpy as np

#data1 = dp.get_data(10, 0, 'train')
pairs, labels = dp.get_mnist_test_data()
print(len(labels))
print(pairs[0][0])
print(pairs[0][0].shape)
print(labels[0])

