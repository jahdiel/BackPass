import numpy as _np
from .primitives import *
from backpass.core.grad_map import add_gradient_pair 

def init_diff_funcs():
    '''Method just to initialize the diff_functions module'''
    pass

def diff_relu(a, ans=None, grad=None):
    return grad * (ans > 0).astype(_np.float32),

add_gradient_pair(relu, diff_relu)

