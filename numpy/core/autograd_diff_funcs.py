'''
The MIT License (MIT)

Copyright (c) 2014 by the President and Fellows of Harvard University and Jahdiel Alvarez

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

import numpy as _np
from .wrap_diffs import debroadcasting
from .primitives import *
from backpass.core.grad_map import add_gradient_pair 

def init_diff_funcs():
    '''Method just to initialize the diff_functions module'''
    pass

def diff_dot(a, b, ans=None, grad=None):
    return dot_adjoint_0(a, b, ans, grad), \
            dot_adjoint_1(a, b, ans, grad)

add_gradient_pair(dot, diff_dot)

def dot_adjoint_0(a, b, ans=None, grad=None):
    '''Great help taken from Autograd library'''
    A_ndim, A_dtype = _np.ndim(a), _np.result_type(a)
    B_ndim = _np.ndim(b)
    if B_ndim == 0 or B_ndim == 1 or A_ndim == 0:
        contract_num = max(0, B_ndim - (A_ndim != 0))
        out = _np.tensordot(grad, b, contract_num)
    else:
        out = _np.tensordot(grad, _np.swapaxes(b, -1, -2), B_ndim - 1)
    return _np.asarray(out, dtype=A_dtype)

def dot_adjoint_1(a, b, ans=None, grad=None):
    '''Great help taken from Autograd library'''
    A_ndim = _np.ndim(a)
    B_ndim, B_dtype = _np.ndim(b), _np.result_type(b)
    needs_transpose = B_ndim > 1 and A_ndim != 0
    swap = (lambda x: _np.swapaxes(x, -1, -2)) if needs_transpose else (lambda x: x)
    if A_ndim == 0 or A_ndim == 1 or B_ndim == 0:
        contract_num = max(0, A_ndim - (B_ndim != 0))
        out = swap(_np.tensordot(grad, a, contract_num))
    else:
        out = swap(_np.tensordot(
            grad, a, [range(-A_ndim - B_ndim + 2, -B_ndim + 1), range(A_ndim - 1)]))
    return _np.asarray(out, dtype=B_dtype)

