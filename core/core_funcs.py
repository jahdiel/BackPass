import numpy as _np
import cupy as _cp
from .tensor import Tensor
from .tensor_gpu import TensorGPU
from .grad_map import grad_map

def grad(func, use_GPU=False):
    if func in grad_map:
        return grad_map[func]
    
    def inner(*args, **kwargs):
        t_args = []
        for arg in args:
            if isinstance(arg, Tensor) or isinstance(arg, TensorGPU):
                arg.ref = 0
                t_args.append(arg)

        # Forward pass
        _cp.cuda.Stream.null.synchronize()
        y = func(*args, **kwargs)
        _cp.cuda.Stream.null.synchronize()

        # Validate the output of func is scalar
        check_output(y, func)
        
        #Backward pass
        _cp.cuda.Stream.null.synchronize()
        backpropagation(y, use_GPU)
        _cp.cuda.Stream.null.synchronize()

        return [arg.grad for arg in t_args]

    return inner

def backpropagation(root, use_GPU):
    """Return a pre-order ordering of nodes in the graph."""
    visited = set()
    root.grad = _cp.array([1]) if use_GPU else _np.array([1]) # dy/dy == 1
    def dfs_walk(node):
        try:
            if not (isinstance(node, Tensor) or isinstance(node, TensorGPU)) or node.parents is None: return
            node.ref -= 1
            if (node.ref > 0): return
            node.pass_grads()
            visited.add(node)
            # print(node, node.parents, "\n")
            for parent in node.tensor_parents():
                if not parent in visited:
                    dfs_walk(parent)

        except AttributeError as attr_err:
            print("Error occured in node:", node.func)
            raise Exception('Attribute Error in a layer.') from attr_err
                
    dfs_walk(root)
    return

def check_output(y, func):
    if not isinstance(y, Tensor) and not isinstance(y, TensorGPU):
            raise ValueError('The function {} needs to return a Tensor object'.format(func))
    if y.value.size != 1:
        raise ValueError('Function {} needs to return a scalar value.'.format(func))