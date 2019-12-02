import numpy as _np
from .wrap_diffs import debroadcasting
from .primitives import *
from backpass.core.grad_map import add_gradient_pair 
from .autograd_diff_funcs import init_diff_funcs as autograd_init_funcs

def init_diff_funcs():
    '''Method just to initialize the diff_functions module'''
    pass

def diff_sum(a, ans=None, grad=None):
    return grad * _np.ones_like(a),

add_gradient_pair(sum, diff_sum)

def diff_square(a, ans=None, grad=None):
    return 2 * grad * a,

add_gradient_pair(square, diff_square)

@debroadcasting
def diff_add(a, b, ans=None, grad=None):
    return grad, grad

add_gradient_pair(add, diff_add)

@debroadcasting
def diff_multiply(a, b, ans=None, grad=None):
    return b * grad, a * grad

add_gradient_pair(multiply, diff_multiply)

@debroadcasting
def diff_subtract(a, b, ans=None, grad=None):
    return grad, -1 * grad

add_gradient_pair(subtract, diff_subtract)

def diff_log(a, ans=None, grad=None):
    return grad / a,

add_gradient_pair(log, diff_log)

def diff_log2(a, ans=None, grad=None):
    return grad / a / _np.log(2),

add_gradient_pair(log2, diff_log2)

def diff_log10(a, ans=None, grad=None):
    return grad / a / _np.log(10),

add_gradient_pair(log10, diff_log10)
