''' Implementation of the Stochastic Gradient Descent Algorithm '''
import numpy as _np

def sgd(weights, learning_rate=0.001):
    # print("in:",weights[0][0].value[100][100])
    # print("grad:", weights[0][0].grad[100][100])
    for W, b in weights:
        # Descend gradients
        W.value -= learning_rate * W.grad
        b.value -= learning_rate * b.grad
        # Zero grads
        W.grad, b.grad = None, None
    # print("in:",weights[0][0].value[100][100])
    return weights