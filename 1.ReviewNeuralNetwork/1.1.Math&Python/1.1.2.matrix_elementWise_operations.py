import numpy as np

W = np.array([[1, 2, 3], [4, 5, 6]])
X = np.array([[0, 1, 2], [3, 4, 5]])

W + X

'''
array([[ 1,  3,  5],
       [ 7,  9, 11]])
'''


W * X

'''
array([[ 0,  2,  6],
       [ 12,  20, 30]])
'''