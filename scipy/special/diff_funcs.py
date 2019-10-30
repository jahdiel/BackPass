import numpy as _np
from .primitives import *
from backpass.core.grad_map import add_gradient_pair 

def init_diff_funcs():
    '''Method just to initialize the diff_functions module'''
    pass


def diff_softmax(a, ans=None, grad=None):
    '''Only works for vectors right now. 
       TODO: Make it function in > 1-D '''
    if _np.ndim(ans) > 1:
        raise ValueError("Only accepts 1-D tensors. Shape: {} not accepted".format(_np.shape(ans)))
    S = _np.tile(ans, (ans.size, 1))
    jacobian = S * (_np.eye(ans.size) - S)
    return _np.dot(grad, jacobian),

add_gradient_pair(softmax, diff_softmax)