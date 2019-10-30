# Backpass - Automatic differentiation with CUDA/Numba support

An automatic differentiation library in Python with CUDA/Numba support. It is based on Tensor objects wich hold their 
gradients once a function is reverse-mode autodifferentiated (backpropagated). The purpose of the library is for beginners 
to better understand the underlying principles of the backpropagation algorithm, the backbone of the deep learning revolution.

Example usage of the library:


```python
import sys
sys.path.append('./')
from backpass.core import grad
import backpass.numpy as np

x = np.array([1, 3, 4, 5, 6], dtype=np.float32)
b = np.array([2, 4, 5, 8, 9], dtype=np.float32)

def s(x, b):
    a = np.square(x)
    c = a * b
    z = a + c
    return np.sum(z)

y = s(x, b)

print('The value of the function:',y.value)

d_func = grad(s)
dy = d_func(x, b)

print('The gradient of the function:', dy)

```

This code results in:

```
The value of the function: 729.0
The gradient of the function: [array([  6.,  30.,  48.,  90., 120.]), array([ 1.,  9., 16., 25., 36.])]
```
