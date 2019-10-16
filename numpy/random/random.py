import numpy.random as _random
from backpass.core import Tensor

def normal(loc=0.0, scale=1.0, *args, **kwargs):
    if isinstance(loc, Tensor.Tensor): loc = loc.value
    if isinstance(scale, Tensor.Tensor): scale = scale.value
    return Tensor.Tensor(_random.normal(loc, scale, *args, **kwargs))

def rand(*args, **kwargs):
    return Tensor.Tensor(_random.rand(*args, **kwargs))