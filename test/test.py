import sys
sys.path.append('../')
from backpass.core import grad
import backpass.numpy as np

x = np.array([1, 3, 4, 5, 6], dtype=np.float32)

b = np.array([2, 4, 5, 8, 9], dtype=np.float32)

def s(x, b):
    a = np.square(x)
    c = np.multiply(a, b)
    z = np.add(a, c)
    return np.sum(z)

y = s(x, b)

print(y.value)

d_func = grad(s)

# print(type(d_func))

dy = d_func(x, b)

print(dy)

# [<backpass.core.Tensor.Tensor 34 >, <backpass.core.Tensor.Tensor [ 2  5 10 17] >, array([1, 1, 1, 1]), <backpass.core.Tensor.Tensor [ 1  4  9 16] >, array([1, 2, 3, 4])]