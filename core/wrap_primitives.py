from functools import wraps
from .tensor import Tensor
from .tensor_gpu import TensorGPU
from backpass.core.grad_map import grad_map

# CPU Primitive functions
def primitive(func):
    ''' Higher order function which wraps the simple numer functions by unboxing/boxing and creating Tensors.
        The wrapper function maintains track of the reference count and add nodes to the computational graph.'''
    @wraps(func)
    def wrapper(*args, **kwargs):
        parents, val_parents = set_parents(*args)
        val_out = func(*val_parents, **kwargs)
        grad_func = grad_map[wrapper] if wrapper in grad_map else None
        return Tensor(val_out, wrapper, parents, grad_func, kwargs)

    return wrapper

# Set parents to the output Tensor
def set_parents(*args):
    p = list(args)
    parents = p
    val_parents = []
    for i in range(len(p)):
        val = p[i]
        if isinstance(p[i], Tensor):
            p[i].ref += 1
            val = p[i].value
        val_parents.append(val)

    return parents, val_parents


# GPU Primitive functions
def gpu_primitive(func):
    ''' Higher order function which wraps the simple numer functions by unboxing/boxing and creating Tensors.
        The wrapper function maintains track of the reference count and add nodes to the computational graph.'''
    @wraps(func)
    def wrapper(*args, **kwargs):
        parents, val_parents = gpu_set_parents(*args)
        val_out = func(*val_parents, **kwargs)
        grad_func = grad_map[wrapper] if wrapper in grad_map else None
        return TensorGPU(val_out, wrapper, parents, grad_func, kwargs)

    return wrapper

# Set parents to the output Tensor
def gpu_set_parents(*args):
    p = list(args)
    parents = p
    val_parents = []
    for i in range(len(p)):
        val = p[i]
        if isinstance(p[i], TensorGPU):
            p[i].ref += 1
            val = p[i].value
        val_parents.append(val)

    return parents, val_parents