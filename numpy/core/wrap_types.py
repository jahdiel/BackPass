from functools import wraps
import numpy as np
from backpass.core import Tensor

# Numpy Types and Structure Wrapping

int32 = np.int32
float32 = np.float32

def array(obj, *args, **kwargs):
    return Tensor(np.array(obj, *args, **kwargs))

def arange(*args, **kwargs):
    return Tensor(np.arange(*args, **kwargs))
