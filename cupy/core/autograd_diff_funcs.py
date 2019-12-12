'''
The MIT License (MIT)

Copyright (c) 2014 by the President, Fellows of Harvard University and Jahdiel Alvarez

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

import cupy as _cp
import numpy as _np
from .wrap_diffs import debroadcasting
import backpass.cupy.core.primitives as cp
from backpass.core.grad_map import add_gradient_pair, grad_map

def init_diff_funcs():
    '''Method just to initialize the diff_functions module'''
    pass

def diff_sum(a, ans=None, grad=None, axis=None, keepdims=False, dtype=None):
    shape, dtype = a.shape, a.dtype
    return repeat_to_match_shape(grad, shape, dtype, axis, keepdims)[0].value,

add_gradient_pair(cp.sum, diff_sum)


def diff_dot(a, b, ans=None, grad=None):
    return dot_adjoint_0(a, b, ans, grad), \
            dot_adjoint_1(a, b, ans, grad)

add_gradient_pair(cp.dot, diff_dot)

def dot_adjoint_0(a, b, ans=None, grad=None):
    '''Great help taken from Autograd library'''
    A_ndim, A_dtype = a.ndim, a.dtype
    B_ndim = b.ndim
    if B_ndim == 0 or B_ndim == 1 or A_ndim == 0:
        contract_num = max(0, B_ndim - (A_ndim != 0))
        out = _cp.tensordot(grad, b, contract_num)
    else:
        out = _cp.tensordot(grad, _cp.swapaxes(b, -1, -2), B_ndim - 1)
    return _cp.asarray(out, dtype=A_dtype)

def dot_adjoint_1(a, b, ans=None, grad=None):
    '''Great help taken from Autograd library'''
    A_ndim = a.ndim
    B_ndim, B_dtype = b.ndim, b.dtype
    needs_transpose = B_ndim > 1 and A_ndim != 0
    swap = (lambda x: _cp.swapaxes(x, -1, -2)) if needs_transpose else (lambda x: x)
    if A_ndim == 0 or A_ndim == 1 or B_ndim == 0:
        contract_num = max(0, A_ndim - (B_ndim != 0))
        out = swap(_cp.tensordot(grad, a, contract_num))
    else:
        out = swap(_cp.tensordot(
            grad, a, [range(-A_ndim - B_ndim + 2, -B_ndim + 1), range(A_ndim - 1)]))
    
    return _cp.asarray(out, dtype=B_dtype)

def diff_reshape(a, newshape, order=None, ans=None, grad=None):
    return cp.reshape(grad, a.shape, order=order)

add_gradient_pair(cp.reshape, diff_reshape)

def repeat_to_match_shape(g, shape, dtype, axis, keepdims):
    """Returns the array g repeated along axis to fit vector space vs.
       Also returns the number of repetitions of the array."""
    if shape == ():
      return g, 1
    axis = list(axis) if isinstance(axis, tuple) else axis
    new_shape = _np.array(shape)
    new_shape[axis] = 1
    num_reps = _np.prod(_np.array(shape)[axis])
    return cp.reshape(g, new_shape) + _cp.zeros(shape, dtype=dtype), num_reps

def diff_mean(a, ans=None, grad=None, axis=None, keepdims=False):
    shape, dtype = a.shape, a.dtype
    g_repeated, num_reps = repeat_to_match_shape(grad, shape, dtype, axis, keepdims)
    return g_repeated.value / num_reps,

add_gradient_pair(cp.mean, diff_mean)

def grad_chooser(a, ans=None, grad=None, axis=None, keepdims=False):
    """Builds gradient of functions that choose a single item, such as min or max."""
    shape, dtype = a.shape, a.dtype
    g_repeated, _ = repeat_to_match_shape(grad, shape, dtype, axis, keepdims)
    argmax_locations = a == repeat_to_match_shape(ans, shape, dtype, axis, keepdims)[0].value
    return g_repeated.value * argmax_locations / _cp.sum(argmax_locations, axis=axis, keepdims=True),
    
add_gradient_pair(cp.max, grad_chooser)
add_gradient_pair(cp.min, grad_chooser)
add_gradient_pair(cp.amax, grad_chooser)
add_gradient_pair(cp.amin, grad_chooser)

def balanced_eq(x, z, y):
    return (x == z) / (1.0 + (x == y))

@debroadcasting
def diff_maximum(a, b, ans=None, grad=None):
    return grad * balanced_eq(a, ans, b), grad * balanced_eq(b, ans, a)

add_gradient_pair(cp.maximum, diff_maximum)