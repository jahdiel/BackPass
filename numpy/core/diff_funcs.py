import numpy as _np

from .primitives import *
from .grad_map import add_gradient_pair 

def init_diff_funcs():
    '''Method just to initialize the diff_functions module'''
    pass

def diff_sum(a, grad):
    return np.array([grad * _np.ones_like(a)])

add_gradient_pair(sum, diff_sum)

def diff_square(a, grad):
    return np.array([2 * grad * a])

add_gradient_pair(square, diff_square)

def diff_add(a, b, grad):
    return np.array([grad, grad])

add_gradient_pair(add, diff_add)

def diff_multiply(a, b, grad):
    return np.array([b * grad, a * grad])

add_gradient_pair(multiply, diff_multiply)

def diff_subtract(a, b, grad):
    return np.array([grad, -1 * grad])

add_gradient_pair(subtract, diff_subtract)