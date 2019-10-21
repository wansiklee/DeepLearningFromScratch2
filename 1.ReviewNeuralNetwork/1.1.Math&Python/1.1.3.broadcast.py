import numpy as np

A = np.array([[1, 2], [3, 4]])
A * 10

'''
array([[ 10,  20],
       [ 30,  40]])
'''


b = np.array([10, 20])
A * b

'''
array([[ 10,  40],
       [ 30,  80]])
'''