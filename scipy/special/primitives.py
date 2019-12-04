import scipy.special as sp
from backpass.core.wrap_primitives import primitive

@primitive
def softmax(a):
    return sp.softmax(a)

@primitive
def logsumexp(a, axis=None, b=None, keepdims=False, return_sign=False):
    return sp.logsumexp(a, axis, b, keepdims, return_sign)