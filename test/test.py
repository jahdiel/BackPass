import sys
sys.path.append('../')
import numpy as _np
from backpass.core import grad
import backpass.numpy as np

x = np.array([1, 2, 3, 4])

z = np.array([2, 3, 1, 1])

print(_np.add(x, z))

def s(a):
    x = np.square(a)
    return np.sum(x)

y = s(x)

print(y)

d_func = grad(s)

print(type(d_func))

dy = d_func(x)

print(dy)