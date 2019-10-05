import numpy as _np
from . import Tensor
from backpass.numpy import grad_map


def grad(func):
    if func in grad_map:
        return grad_map[func]
    
    def inner(*args, **kwargs):
        for arg in args:
            if isinstance(arg, Tensor):
                arg.ref = 0
        # print('Args:',args)
        # Forward pass
        y = func(*args, **kwargs)
        if not isinstance(y, Tensor):
            raise ValueError('The function {} needs to return a Tensor object'.format(func))
        if y.value.size != 1:
            raise ValueError('Function {} needs to return a scalar value.'.format(func))
        
        #Backward pass
        backpropagation(y)

        return [arg.grad for arg in args]

    return inner

def backpropagation(root):
    """Return a post-order ordering of nodes in the graph."""
    visited = set()
    root.grad = _np.array([1]) # dy/dy == 1
    def dfs_walk(node):
        if not isinstance(node, Tensor) or node.parents is None: return
        node.ref -= 1
        if (node.ref > 0): return
        node.pass_grads()
        visited.add(node)
        # print(node, node.grad)
        for succ in node.parents:
            if not succ in visited:
                dfs_walk(succ)
                
    dfs_walk(root)
    return