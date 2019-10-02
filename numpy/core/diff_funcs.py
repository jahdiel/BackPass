import numpy as _np

from .funcs import *
from .grad_map import add_gradient_pair 

def init_diff_funcs():
    '''Method just to initialize the diff_functions module'''
    pass

def diff_sum(a):
    return _np.ones_like(a)

add_gradient_pair(sum, diff_sum)

def diff_square(a):
    return 2 * a

add_gradient_pair(square, diff_square)

def diff_add(a, b):
    if a.shape != b.shape:
        raise ValueError('The shape of {} and {} dont match. ({} != {})'.format(a, b, a.shape, b.shape)) 
    max_dim = max(a.shape)

    return np.eye(max_dim)

add_gradient_pair(add, diff_add)