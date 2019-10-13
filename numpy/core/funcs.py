import numpy as np
from backpass.core import Tensor
from backpass.numpy.core.grad_map import grad_map

# Numpy Types and Structure Wrapping

int32 = np.int32
float32 = np.float32

def array(obj, *args, **kwargs):
    return Tensor(np.array(obj, *args, **kwargs))

# Primitive functions

def primitive(func):
    ''' Higher order function which wraps the simple numer functions by unboxing/boxing and creating Tensors.
        The wrapper function maintains track of the reference count and add nodes to the computational graph.'''
    def wrapper(*args):
        parents, val_parents = set_parents(*args)
        val_out = func(*val_parents)
        return Tensor(val_out, wrapper, parents, grad_map[wrapper])

    return wrapper


@primitive
def sum(a):
    return np.sum(a)

@primitive
def square(a):
    return np.square(a)

@primitive
def add(a, b):
    if a.shape != b.shape:
        raise ValueError('The shape of {} and {} dont match. ({} != {})'.format(a, b, a.shape, b.shape))
    return np.add(a, b)

@primitive
def multiply(a, b):
    if a.shape != b.shape:
        raise ValueError('The shape of {} and {} dont match. ({} != {})'.format(a, b, a.shape, b.shape))
    return np.multiply(a, b)


def set_parents(*args):
    parents = list(args)
    val_parents = []
    for i in range(len(parents)):
        val = parents[i]
        if isinstance(parents[i], Tensor):
            parents[i].ref += 1
            val = parents[i].value
        val_parents.append(val)

    return parents, val_parents

