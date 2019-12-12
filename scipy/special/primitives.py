import scipy.special as _sp
import cupy as _cp
from backpass.core.wrap_primitives import primitive, gpu_primitive

@primitive
def softmax(a):
    return _sp.softmax(a)

@primitive
def logsumexp(a, axis=None, b=None, keepdims=False, return_sign=False):
    return _sp.logsumexp(a, axis, b, keepdims, return_sign)

@gpu_primitive
def gpu_logsumexp(a, axis=None, b=None, keepdims=False, return_sign=False):
    return _cp.array(_sp.logsumexp(a.get(), axis, b, keepdims, return_sign))