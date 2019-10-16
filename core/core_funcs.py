import numpy as _np
from . import Tensor
from backpass.numpy import grad_map


def grad(func):
    if func in grad_map:
        return grad_map[func]
    
    def inner(*args, **kwargs):
        t_args = []
        for arg in args:
            if isinstance(arg, Tensor):
                arg.ref = 0
                t_args.append(arg)

        # Forward pass
        y = func(*args, **kwargs)
        
        # Validate the output of func
        check_output(y, func)
        
        #Backward pass
        backpropagation(y)

        return [arg.grad for arg in t_args]

    return inner

def backpropagation(root):
    """Return a pre-order ordering of nodes in the graph."""
    visited = set()
    root.grad = _np.array([1]) # dy/dy == 1
    def dfs_walk(node):
        if not isinstance(node, Tensor) or node.parents is None: return
        node.ref -= 1
        if (node.ref > 0): return
        node.pass_grads()
        visited.add(node)
        # print(node, node.grad)
        for parent in node.parents:
            if not parent in visited:
                dfs_walk(parent)
                
    dfs_walk(root)
    return

def check_output(y, func):
    if not isinstance(y, Tensor):
            raise ValueError('The function {} needs to return a Tensor object'.format(func))
    if y.value.size != 1:
        raise ValueError('Function {} needs to return a scalar value.'.format(func))