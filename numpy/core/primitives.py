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
def mean(a, axis=None, dtype=None, out=None, keepdims=False):
    return _np.mean(a, axis, dtype, out, keepdims)

@primitive
def reshape(a, newshape, order='C'):
    return _np.reshape(a, newshape, order)

@primitive
def argmax(a, axis=None, out=None):
    return _np.argmax(a, axis, out)

@primitive
def maximum(a, b, out=None, where=True, casting='same_kind', order='K', dtype=None, subok=True):
    return _np.maximum(a, b, out=out, where=where, casting=casting, order=order, dtype=dtype, subok=subok)

@primitive
def max(a, axis=None, out=None, keepdims=False, initial=_np._NoValue, where=True): 
    return _np.max(a, axis, out, keepdims, initial, where)

@primitive
def amax(a, axis=None, out=None, keepdims=_np._NoValue, initial=_np._NoValue, where=_np._NoValue): 
    return _np.amax(a, axis, out, keepdims, initial, where)

@primitive
def min(a, axis=None, out=None, keepdims=False, initial=_np._NoValue, where=True): 
    return _np.min(a, axis, out, keepdims, initial, where)

@primitive
def amin(a, axis=None, out=None, keepdims=_np._NoValue, initial=_np._NoValue, where=_np._NoValue): 
    return _np.amin(a, axis, out, keepdims, initial, where)

@primitive
def equal(a, b):
    return _np.equal(a, b)
