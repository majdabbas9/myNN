import matplotlib.pyplot as plt
import nnfs
from nnfs.datasets import vertical_data
from backend import np
# Passed in gradient from the next layer
# for the purpose of this example we're going to use
# a vector of 1sb n  
arr = np.array([[10,15,-20,-1],[1,2,3,4]])
arr2 = np.array([[2,2,2,2],[2,2,2,1]])
print(arr / arr2)