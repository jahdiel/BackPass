from .tensor import Tensor
from .tensor_gpu import TensorGPU
from .core_funcs import grad
from .wrap_primitives import primitive, gpu_primitive
from .grad_map import grad_map

def init_grad_map():
    from backpass.numpy.core.diff_funcs import init_diff_funcs
    from backpass.scipy.special.diff_funcs import init_diff_funcs
    from backpass.nn.core.diff_funcs import init_diff_funcs
    from backpass.cupy.core.diff_funcs import init_diff_funcs
    return

init_grad_map()