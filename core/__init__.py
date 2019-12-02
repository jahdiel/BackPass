from .tensor import Tensor
from .core_funcs import grad
from .wrap_primitives import primitive
from .grad_map import grad_map

def init_grad_map():
    from backpass.numpy.core.diff_funcs import init_diff_funcs
    from backpass.scipy.special.diff_funcs import init_diff_funcs
    from backpass.nn.core.diff_funcs import init_diff_funcs
    return

init_grad_map()