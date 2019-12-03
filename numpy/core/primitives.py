import numpy as _np
from backpass.core.wrap_primitives import primitive

@primitive
def negative(a):
    return _np.negative(a)

@primitive
def sum(a):
    return _np.sum(a)

@primitive
def square(a):
    return _np.square(a)

@primitive
def log(a):
    return _np.log(a)

@primitive
def log2(a):
    return _np.log2(a)

@primitive
def log10(a):
    return _np.log10(a)

@primitive
def add(a, b):
    return _np.add(a, b)

@primitive
def subtract(a, b):
    return _np.subtract(a, b)

@primitive
def multiply(a, b):
    return _np.multiply(a, b)

@primitive
def divide(a, b):
    return _np.divide(a, b)

@primitive
def true_divide(a, b):
    return _np.true_divide(a, b)

@primitive
def dot(a, b):
    return _np.dot(a, b)

@primitive
def argmax(a, axis=None, out=None):
    return _np.argmax(a, axis, out)

@primitive
def mean(a, axis=None, dtype=None, out=None, keepdims=False):
    return _np.mean(a, axis, dtype, out, keepdims)

@primitive
def reshape(a, newshape, order='C'):
    return _np.reshape(a, newshape, order)


