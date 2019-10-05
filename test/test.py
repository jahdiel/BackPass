import sys
sys.path.append('../')
from backpass.core import grad
import backpass.numpy as np

x = np.array([1])

b = np.array([2])

def s(x, b):
    a = np.square(x)
    c = np.multiply(a, b)
    return np.add(a, c)


y = s(x, b)

print('Result:', y)

d_func = grad(s)

# print(type(d_func))

dy = d_func(x, b)

print('Grads:', dy)

# [<backpass.core.Tensor.Tensor 34 >, <backpass.core.Tensor.Tensor [ 2  5 10 17] >, array([1, 1, 1, 1]), <backpass.core.Tensor.Tensor [ 1  4  9 16] >, array([1, 2, 3, 4])]