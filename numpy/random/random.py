import numpy.random as _random
from backpass.core import Tensor

def normal(loc=0.0, scale=1.0, *args, **kwargs):
    if isinstance(loc, Tensor): loc = loc.value
    if isinstance(scale, Tensor): scale = scale.value
    return Tensor(_random.normal(loc, scale, *args, **kwargs))

def rand(*args, **kwargs):
    return Tensor(_random.rand(*args, **kwargs))

def randn(*args, **kwargs):
    return Tensor(_random.randn(*args, **kwargs))