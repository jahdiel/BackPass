import cupy.random as _random
from backpass.core import TensorGPU

def normal(loc=0.0, scale=1.0, *args, **kwargs):
    if isinstance(loc, TensorGPU): loc = loc.value
    if isinstance(scale, TensorGPU): scale = scale.value
    return TensorGPU(_random.normal(loc, scale, *args, **kwargs))

def rand(*args, **kwargs):
    return TensorGPU(_random.rand(*args, **kwargs))

def randn(*args, **kwargs):
    return TensorGPU(_random.randn(*args, **kwargs))