import numpy as _np
from .wrap_diffs import debroadcasting
from .primitives import *
from .grad_map import add_gradient_pair 

def init_diff_funcs():
    '''Method just to initialize the diff_functions module'''
    pass

def diff_sum(a, grad=None):
    return grad * _np.ones_like(a),

add_gradient_pair(sum, diff_sum)

def diff_square(a, grad=None):
    return 2 * grad * a,

add_gradient_pair(square, diff_square)

@debroadcasting
def diff_add(a, b, grad=None):
    return grad, grad

add_gradient_pair(add, diff_add)

@debroadcasting
def diff_multiply(a, b, grad=None):
    return b * grad, a * grad

add_gradient_pair(multiply, diff_multiply)

@debroadcasting
def diff_subtract(a, b, grad=None):
    return grad, -1 * grad

add_gradient_pair(subtract, diff_subtract)

