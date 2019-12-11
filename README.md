<p align=center>
    <img src="https://github.com/jahdiel/backpass/blob/master/Backpass.png" width=300 height=250>
</p>

# Backpass - Automatic differentiation with CUDA

An automatic differentiation library in Python with CUDA support. It is based on Tensor objects which construct a computational graph used to obtained the gradient of a function using reverse-mode autodifferentiation (backpropagation). The purpose of the library is for beginners to better understand the underlying principles of the backpropagation algorithm, the backbone of the deep learning revolution.

Backpass is wrapps numpy and scipy functions, although not all functions are implemented. Native python, numpy and/or scipy code can be differentiated using the library, there is no need of specialized sub-languages to construct the graphs. We use a dynamic approach to the creation of the computational graphs. 

The advantage of backpass is that it supports CUDA making it much faster than other autodiff packages in python and that it is very lightweight in contrast to other established deep learning frameworks. 

Example usage of the library:


```python
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
