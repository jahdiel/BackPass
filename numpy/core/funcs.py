import numpy as np
from backpass.core import Tensor
from backpass.numpy.core.grad_map import grad_map

def array(obj, *args, **kwargs):
    return Tensor(np.array(obj, *args, **kwargs))
   
def sum(a):
    parents, (val_a,) = set_parents(a)

    return Tensor(np.sum(val_a), sum, parents, grad_map[sum])

def square(a):
    parents, (val_a,) = set_parents(a)

    return Tensor(np.square(val_a), square, parents, grad_map[square])

def add(a, b):
    parents, (val_a, val_b) = set_parents(a, b)

    if val_a.shape != val_b.shape:
        raise ValueError('The shape of {} and {} dont match. ({} != {})'.format(val_a, val_b, val_a.shape, val_b.shape))

    val_y = np.add(val_a, val_b)

    return Tensor(val_y, add, parents, grad_map[add])

def primitive(func):
    def wrapper(*args):
        parents, val_parents = set_parents(*args)
        val_out = func(*val_parents)
        return Tensor(val_out, func, parents, grad_map[func])

    return wrapper

@primitive
def multiply(a, b):
    if a.shape != b.shape:
        raise ValueError('The shape of {} and {} dont match. ({} != {})'.format(a, b, a.shape, b.shape))
    return np.multiply(a, b)

# def multiply(a, b):
#     parents, val_parents = set_parents(a, b)
    
#     if val_parents[0].shape != val_parents[1].shape:
#         raise ValueError('The shape of {} and {} dont match. ({} != {})'.format(val_parents[0], val_parents[1], val_parents[0].shape, val_parents[1].shape))

#     val_out = np.multiply(val_parents[0], val_parents[1])

#     return Tensor(val_out, multiply, parents, grad_map[multiply])

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

