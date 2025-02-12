import numpy as np
from numba import jit


@jit
def reorder_cycle_sort(arr, index):
    for i in range(len(arr)):
        if index[i] >= 0:
            start = i
            val = arr[i]
            while index[i] != start:
              arr[i] = arr[index[i]]
              new_i = index[i]
              index[i] = -1
              i = new_i
              #index[i], i = -1, index[i]  
            arr[i] = val
            index[i] = -1
                


rng = np.random.default_rng(42)

arr = rng.permutation(10**8)

ind = arr.argsort()

reorder_cycle_sort(arr, ind)
