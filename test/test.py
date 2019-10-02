import sys
sys.path.append('../')
import numpy as _np
from backpass.core import grad
import backpass.numpy as np

x = np.array([1, 2, 3, 4])

z = np.array([2, 3, 1, 1])

print(_np.add(x, z))

def s(a, b):
    x = np.square(a)
    y = np.add(x, b)
    return np.sum(y)

y = s(x, z)

print(y)

d_func = grad(s)

print(type(d_func))

dy = d_func(x, z)

print(dy)

# [<backpass.core.Tensor.Tensor 34 >, <backpass.core.Tensor.Tensor [ 2  5 10 17] >, array([1, 1, 1, 1]), <backpass.core.Tensor.Tensor [ 1  4  9 16] >, array([1, 2, 3, 4])]