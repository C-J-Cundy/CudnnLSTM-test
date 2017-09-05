import numpy as np
from cudnn_func import cudnn

iter_list = []
times_list = []

for _ in range(5):
    a, b = cudnn(1024, 1024, 8, 2)
    iter_list.append(a)
    times_list.append(b)

print iter_list, times_list
