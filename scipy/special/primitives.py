import scipy.special as sp
from backpass.core.wrap_primitives import primitive

@primitive
def softmax(a):
    return sp.softmax(a)