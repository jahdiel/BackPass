import numpy as np
from backpass.core import Tensor
from backpass.numpy.core.grad_map import grad_map

# Numpy Types and Structure Wrapping

int32 = np.int32
float32 = np.float32

def array(obj, *args, **kwargs):
    return Tensor.Tensor(np.array(obj, *args, **kwargs))

def arange(*args, **kwargs):
    return Tensor.Tensor(np.arange(*args, **kwargs))

# Primitive functions

def primitive(func):
    ''' Higher order function which wraps the simple numer functions by unboxing/boxing and creating Tensors.
        The wrapper function maintains track of the reference count and add nodes to the computational graph.'''
    def wrapper(*args):
        parents, val_parents = set_parents(*args)
        val_out = func(*val_parents)
        return Tensor.Tensor(val_out, wrapper, parents, grad_map[wrapper])

    return wrapper


def set_parents(*args):
    p = list(args)
    parents = p
    val_parents = []
    for i in range(len(p)):
        val = p[i]
        if isinstance(p[i], Tensor.Tensor):
            p[i].ref += 1
            val = p[i].value
        val_parents.append(val)

    return parents, val_parents

