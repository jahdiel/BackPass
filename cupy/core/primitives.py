import cupy as _cp
from backpass.core.wrap_primitives import gpu_primitive

@gpu_primitive
def negative(a):
    return _cp.negative(a)

@gpu_primitive
def sum(a, axis=None, dtype=None, out=None):
    return _cp.sum(a, axis=axis, dtype=dtype, out=out)

@gpu_primitive
def square(a):
    return _cp.square(a)

@gpu_primitive
def log(a):
    return _cp.log(a)

@gpu_primitive
def log2(a):
    return _cp.log2(a)

@gpu_primitive
def log10(a):
    return _cp.log10(a)

@gpu_primitive
def add(a, b):
    return _cp.add(a, b)

@gpu_primitive
def subtract(a, b):
    return _cp.subtract(a, b)

@gpu_primitive
def multiply(a, b):
    return _cp.multiply(a, b)

@gpu_primitive
def divide(a, b):
    return _cp.divide(a, b)

@gpu_primitive
def true_divide(a, b):
    return _cp.true_divide(a, b)

@gpu_primitive
def dot(a, b):
    return _cp.dot(a, b)

@gpu_primitive
def mean(a, axis=None, dtype=None, out=None, keepdims=False):
    return _cp.mean(a, axis, dtype, out, keepdims)

@gpu_primitive
def reshape(a, newshape, order='C'):
    return _cp.reshape(a, newshape, order)

@gpu_primitive
def argmax(a, axis=None, dtype=None, out=None, keepdims=False):
    return _cp.argmax(a, axis=axis, dtype=dtype, out=out, keepdims=keepdims)

@gpu_primitive
def maximum(a, b, out=None, where=True, casting='same_kind', order='K', dtype=None, subok=True):
    return _cp.maximum(a, b, out=out, where=where, casting=casting, order=order, dtype=dtype, subok=subok)

@gpu_primitive
def max(a, axis=None, out=None, keepdims=False, where=True): 
    return _cp.max(a, axis, out, keepdims, where=where)

@gpu_primitive
def amax(a, axis=None, out=None): 
    return _cp.amax(a, axis, out)

@gpu_primitive
def min(a, axis=None, out=None, keepdims=False, where=True): 
    return _cp.min(a, axis, out, keepdims, where=where)

@gpu_primitive
def amin(a, axis=None, out=None): 
    return _cp.amin(a, axis, out)

@gpu_primitive
def equal(a, b):
    return _cp.equal(a, b)

@gpu_primitive
def tanh(x, out=None, casting='same_kind', dtype=None):
    return _cp.tanh(x, out=out, casting=casting, dtype=dtype)
