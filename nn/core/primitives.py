import numpy as np
from backpass.core.wrap_primitives import primitive

@primitive
def relu(a):
    return np.maximum(0, a)
