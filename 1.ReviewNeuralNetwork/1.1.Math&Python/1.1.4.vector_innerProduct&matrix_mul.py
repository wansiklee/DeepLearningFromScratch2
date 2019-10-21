import numpy as np

# 백터의 내적
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
np.dot(a, b)

'''
32
'''


# 행렬의 곱
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
np.matmul(A, B)

'''
array([[ 19,  22],
       [ 43,  50]])
'''