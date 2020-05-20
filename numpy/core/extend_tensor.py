import backpass.numpy.core.primitives as primitives
from backpass.core import Tensor


class NPTensor:
    """
    Class to extend the Tensor object with numpy numerical methods. 
    The methods in this class are passed to the Tensor object by setting
    them as attributes.
    """
    def __neg__(self):
        return primitives.negative(self)

    def __eq__(self, other):
        return primitives.equal(self, other)

    def __add__(self, other):
        return primitives.add(self, other)

    def __radd__(self, other):
        return primitives.add(other, self)

    def __mul__(self, other):
        return primitives.multiply(self, other)
    
    def __rmul__(self, other):
        return primitives.multiply(other, self)

    def __sub__(self, other):
        return primitives.subtract(self, other)

    def __rsub__(self, other):
        return primitives.subtract(other, self)

    def __div__(self, other):
        return primitives.divide(self, other)
    
    def __rdiv__(self, other):
        return primitives.divide(other, self)

    def __truediv__(self, other):
        return primitives.true_divide(self, other)
    
    def __rtruediv__(self, other):
        return primitives.true_divide(other, self)

# NPTensor attributes which should not be passed to the Tensor object
reserved = set(['__module__', '__dict__', '__weakref__', '__doc__', '__hash__'])

# Add some numpy numerical functions to the Tensor object
for func_name in NPTensor.__dict__:
    if func_name not in reserved:
        func = NPTensor.__dict__[func_name]
        setattr(Tensor, func.__name__, func)