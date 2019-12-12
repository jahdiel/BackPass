from functools import wraps
import cupy as cp
from backpass.core import TensorGPU

# Numpy Types and Structure Wrapping

int32 = cp.int32
float32 = cp.float32

def array(obj, *args, **kwargs):
    return TensorGPU(cp.array(obj, *args, **kwargs))

def arange(*args, **kwargs):
    return TensorGPU(cp.arange(*args, **kwargs))
