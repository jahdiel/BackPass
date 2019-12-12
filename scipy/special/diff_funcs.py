import numpy as _np
import cupy as _cp
from .primitives import *
from backpass.core.grad_map import add_gradient_pair
import backpass.numpy.core.autograd_diff_funcs as np_autodiff
import backpass.cupy.core.autograd_diff_funcs as cp_autodiff

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

def diff_logsumexp(a, ans=None, grad=None, axis=None, b=1.0, keepdims=False):
    shape, dtype = _np.shape(a), _np.result_type(a)
    g_repeated,   _ = np_autodiff.repeat_to_match_shape(grad, shape, dtype, axis, keepdims)
    ans_repeated, _ = np_autodiff.repeat_to_match_shape(ans, shape, dtype, axis, keepdims)
    # print("a:", a)
    # print("ans:", ans)
    # print("g_repeated:", g_repeated)
    # print("ans_repeated:", ans_repeated)
    # print("res:", g_repeated.value * b * _np.exp(a - ans_repeated.value))
    return g_repeated.value * b * _np.exp(a - ans_repeated.value)

add_gradient_pair(logsumexp, diff_logsumexp)

def diff_gpu_logsumexp(a, ans=None, grad=None, axis=None, b=1.0, keepdims=False):
    shape, dtype = a.shape, a.dtype
    g_repeated,   _ = cp_autodiff.repeat_to_match_shape(grad, shape, dtype, axis, keepdims)
    ans_repeated, _ = cp_autodiff.repeat_to_match_shape(ans, shape, dtype, axis, keepdims)
    return g_repeated.value * b * _cp.exp(a - ans_repeated.value)

add_gradient_pair(gpu_logsumexp, diff_gpu_logsumexp)