import cupy as _cp
from backpass.cupy.core.wrap_diffs import debroadcasting
import backpass.cupy.core.primitives as cp
from backpass.core.grad_map import add_gradient_pair 
from .autograd_diff_funcs import init_diff_funcs as autograd_init_funcs

def init_diff_funcs():
    '''Method just to initialize the diff_functions module'''
    pass

def diff_negative(a, ans=None, grad=None):
    return -grad,

add_gradient_pair(cp.negative, diff_negative)

def diff_square(a, ans=None, grad=None):
    return 2 * grad * a,

add_gradient_pair(cp.square, diff_square)

@debroadcasting
def diff_add(a, b, ans=None, grad=None):
    return grad, grad

add_gradient_pair(cp.add, diff_add)

@debroadcasting
def diff_subtract(a, b, ans=None, grad=None):
    return grad, -1 * grad

add_gradient_pair(cp.subtract, diff_subtract)

@debroadcasting
def diff_multiply(a, b, ans=None, grad=None):
    return b * grad, a * grad

add_gradient_pair(cp.multiply, diff_multiply)

@debroadcasting
def diff_divide(a, b, ans=None, grad=None):
    return grad / b, -grad * a / b**2

add_gradient_pair(cp.divide, diff_divide)

@debroadcasting
def diff_true_divide(a, b, ans=None, grad=None):
    return grad / b, -grad * a / b**2

add_gradient_pair(cp.true_divide, diff_true_divide)

def diff_log(a, ans=None, grad=None):
    return grad / a,

add_gradient_pair(cp.log, diff_log)

def diff_log2(a, ans=None, grad=None):
    return grad / a / _cp.log(2),

add_gradient_pair(cp.log2, diff_log2)

def diff_log10(a, ans=None, grad=None):
    return grad / a / _cp.log(10),

add_gradient_pair(cp.log10, diff_log10)

def diff_tanh(a, ans=None, grad=None):
    # print("max a:", _cp.max(a))
    # print("res:", _cp.cosh(_cp.max(a)) **2)
    return grad / _cp.cosh(a) **2,

add_gradient_pair(cp.tanh, diff_tanh)