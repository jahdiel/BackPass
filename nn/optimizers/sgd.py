''' Implementation of the Stochastic Gradient Descent Algorithm '''

def sgd(weights, learning_rate=0.001):
    for W, b in weights:
        # Descend gradients
        W.value -= learning_rate * W.grad
        b.value -= learning_rate * b.grad
        # Zero grads
        W.grad, b.grad = 0, 0

    return weights