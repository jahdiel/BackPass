<p align=center>
    <img src="https://github.com/jahdiel/backpass/blob/master/Backpass.png" width=300 height=250>
</p>

# Backpass - Automatic differentiation with CUDA

An automatic differentiation library in Python with CUDA support. It is based on Tensor objects which construct a computational graph used to obtained the gradient of a function using reverse-mode autodifferentiation (backpropagation). The purpose of the library is for beginners to better understand the underlying principles of the backpropagation algorithm, the backbone of the deep learning revolution.

Backpass wraps numpy and scipy functions, although not all functions are implemented. Native python, numpy and/or scipy code can be differentiated using the library, there is no need of specialized sub-languages to construct the graphs. We use a dynamic approach for the creation of the computational graphs. 

The advantages of backpass are that it supports CUDA, something other autodiff packages in python don't do and it is very lightweight, meaning it can be run on virtually any device. 

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

## Neural Net for MNIST dataset using CuPy

Here is a simple feedforward neural net for solving the MNIST dataset using GPU.

```python
import sys
sys.path.append('../../')
from backpass.core import grad
import backpass.cupy as cp
import backpass.cupy.random as rnp
import backpass.scipy.special as sp
import backpass.nn as nn
from backpass.nn.optimizers import sgd, sgd_momentum
from data import load_mnist
import numpy as _np
import cupy as _cp
sys.version

def init_weights(layers, scale=0.1):
    weights = [(scale * rnp.randn(layers[i], layers[i+1]), 
                scale * rnp.randn(layers[i+1])) 
               for i in range(len(layers[:-1]))]
    for W, b in weights:
        W.parents, b.parents = None, None
        W.ref, b.ref = 0, 0
    return weights

def model(inputs, weights):
    last_idx = len(weights) - 1
    for idx, (W, b) in enumerate(weights):
        outputs = cp.dot(inputs, W) + b
        inputs  = cp.tanh(outputs) if idx != last_idx else outputs - sp.gpu_logsumexp(outputs, axis=1, keepdims=True)
    return inputs

def loss(predictions, label):
    return -cp.sum(predictions * cp.array(label))

def accuracy(inputs, weights, labels):
    target_class    = cp.argmax(labels, axis=1)
    predicted_class = cp.argmax(model(inputs, weights), axis=1)
    return cp.mean(predicted_class == target_class)

def batch_indices(iter, batch_size, num_batches):
    idx = iter % num_batches
    return slice(idx * batch_size, (idx+1) * batch_size)

def objective(x, weights, label):
    preds = model(x, weights)
    return loss(preds, label)

# Gradients of functions
grad_objective = grad(objective, use_GPU=True)

def print_perf(weights, epoch):
    train_acc = accuracy(train_images, weights, train_labels)
    test_acc  = accuracy(test_images, weights, test_labels)
    batch_idx = batch_indices(0, 256, 234)
    loss      = objective(train_images[batch_idx], weights, train_labels[batch_idx])
    print("{:15}|{:20}|{:20}|{:20}".format(epoch, train_acc.value.get(), test_acc.value.get(), loss.value.get()))

print("Loading training data...")
N, train_images, train_labels, test_images, test_labels = load_mnist()

train_images, train_labels = _cp.array(train_images), _cp.array(train_labels)
test_images, test_labels   = _cp.array(test_images), _cp.array(test_labels)

# Hyperparameters
epochs = 10
lr = 0.001
batch_size = 1
weight_scale = 0.1
num_batches = int(_np.ceil(len(train_images) / batch_size))

# Weights
layers = [784, 200, 100, 10]
weights = init_weights(layers, weight_scale)

# Optimizer
run_optimizer = sgd_momentum(weights, learning_rate=lr)

print("     Epoch     |    Train accuracy  |     Test accuracy  |          Loss      |      L2      ")

# Training Loop: Use SGD for gradient descent
for epoch in range(epochs):
    for idx in range(num_batches):
        # Get training batch indices
        batch_idx = batch_indices(idx, batch_size, num_batches)
        # Backpropagation: Get the gradient for all weights
        grad_objective(train_images[batch_idx], weights, train_labels[batch_idx])
        # Print accuracy per epoch
        if idx == 0:    
            print_perf(weights, epoch)
        # Gradient descent errors
        run_optimizer()

print("\nFinished.")

```
