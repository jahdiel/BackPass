import numpy as _np
from . import Tensor
from backpass.numpy import grad_map


def grad(func):
    if func in grad_map:
        return grad_map[func]
    
    def inner(*args, **kwargs):
        print('Args:',args)
        # Forward pass
        y = func(*args, **kwargs)
        if not isinstance(y, Tensor):
            raise ValueError('The function {} needs to return a Tensor object'.format(func))
        if not _np.isscalar(y.value):
            raise ValueError('Function {} needs to return a scalar value.'.format(func))
        
        #Backward pass
        order = reverse_postorder(y)
        print(order)

    return inner

def reverse_postorder(root):
    """Return a post-order ordering of nodes in the graph."""
    visited = set()
    order = []
    def dfs_walk(node):
        visited.add(node)
        if isinstance(node, Tensor):
            for succ in node.parents:
                if isinstance(succ, Tensor):
                    if not succ in visited:
                        dfs_walk(succ)
                else:
                    order.append(succ)
        order.append(node)
    dfs_walk(root)
    order.reverse()
    return order