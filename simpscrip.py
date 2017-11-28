import data_processing as dp 
import numpy as np

#data1 = dp.get_data(10, 0, 'train')
data = dp.get_test_data()
print(len(data))
print('img')
print(data[0][0][0])
print('label')
print(data[1][0])
print(len(data[0]))
print(len(data[1]))
