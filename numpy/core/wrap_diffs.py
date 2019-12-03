from functools import wraps
import numpy as _np

def debroadcasting(diff_func):
    ''' Ensure that the grad for each argument has same shape as argument.
        This means taking care of broadcasted arguments used to evaluate primitives.'''
    @wraps(diff_func)
    def reversed_func(*args, ans=None, grad=None):
        proper_grads = []
        grads = diff_func(*args, grad=grad)
        for arg, grad in zip(args, grads):
            proper_grads.append(debroadcast(arg, grad))
        return tuple(proper_grads)
    reversed_func.__name__ = diff_func.__name__
    return reversed_func

def debroadcast(arg, grad, debroadcast_axis=0):
    arg_dim = _np.ndim(arg)
    while _np.ndim(grad) > arg_dim:
        grad = _np.sum(grad, axis=debroadcast_axis)
    for axis, size in enumerate(_np.shape(arg)):
        if size == 1:
            grad = _np.sum(grad, axis=axis, keepdims=True)
    return grad