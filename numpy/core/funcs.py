import numpy as np
from backpass.core import Tensor

def array(obj, *args, **kwargs):
    return np.array(obj, *args, **kwargs)

def sum(a):
    parents = []

    val_a = a
    parents.append(val_a)
    if isinstance(val_a, Tensor):
        val_a = val_a.value

    val_y = np.sum(val_a)

    return Tensor(val_y, sum, parents)

def square(a):
    parents = []

    val_a = a
    parents.append(val_a)
    if isinstance(val_a, Tensor):
        val_a = val_a.value

    val_y = np.square(val_a)

    return Tensor(val_y, square, parents)

def add(a, b):
     
    parents = []

    val_a = a
    val_b = b
    parents.append(val_a)
    parents.append(val_b)
    if isinstance(val_a, Tensor):
        val_a = val_a.value
    if isinstance(val_b, Tensor):
        val_b = val_b.value

    if val_a.shape != val_b.shape:
        raise ValueError('The shape of {} and {} dont match. ({} != {})'.format(val_a, val_b, val_a.shape, val_b.shape))

    val_y = np.add(val_a, val_b)

    return Tensor(val_y, add, parents)