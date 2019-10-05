import numpy as _np

from .funcs import *
from .grad_map import add_gradient_pair 

def init_diff_funcs():
    '''Method just to initialize the diff_functions module'''
    pass

def diff_sum(a, grad):
    return grad * _np.ones_like(a)

add_gradient_pair(sum, diff_sum)

def diff_square(a, grad):
    return np.array(2 * grad * a.value)

add_gradient_pair(square, diff_square)

def diff_add(a, b, grad):
    if a.value.shape != b.value.shape:
        raise ValueError('The shape of {} and {} dont match. ({} != {})'.format(a.value, b.value, a.value.shape, b.value.shape)) 

    return np.array([grad, grad])

add_gradient_pair(add, diff_add)

def diff_multiply(a, b, grad):
    if a.value.shape != b.value.shape:
        raise ValueError('The shape of {} and {} dont match. ({} != {})'.format(a.value, b.value, a.value.shape, b.value.shape))  

    return np.array([b.value, a.value])

add_gradient_pair(multiply, diff_multiply)