from functools import wraps
import cupy as _cp

def debroadcasting(diff_func):
    ''' Ensure that the grad for each argument has same shape as argument.
        This means taking care of broadcasted arguments used to evaluate primitives.'''
    @wraps(diff_func)
    def reversed_func(*args, **kwargs):
        proper_grads = []
        grads = diff_func(*args, **kwargs)
        for arg, grad in zip(args, grads):
            proper_grads.append(debroadcast(arg, grad))
        return tuple(proper_grads)
    reversed_func.__name__ = diff_func.__name__
    return reversed_func

def debroadcast(arg, grad, debroadcast_axis=0):
    arg_dim = arg.ndim
    while grad.ndim > arg_dim:
        grad = _cp.sum(grad, axis=debroadcast_axis)
    for axis, size in enumerate(arg.shape):
        if size == 1:
            grad = _cp.sum(grad, axis=axis, keepdims=True)
    return grad