''' Implementation of the Stochastic Gradient Descent Algorithm '''
import numpy as _np


def sgd(weights, learning_rate=0.001):
    def optimizer():
        # print("in:",weights[0][0].value[100][100])
        # print("grad shape:", weights[0][0].grad.shape)
        for W, b in weights:
            # Descend gradients
            W.value -= learning_rate * W.grad
            b.value -= learning_rate * b.grad
            # Zero grads
            W.grad, b.grad = None, None
            # print("ref:", W.ref, b.ref)
            W.ref, b.ref = 0, 0
        # print("in:",weights[0][0].value[100][100])
    return optimizer

    
def sgd_momentum(weights, learning_rate=0.001, beta=0.9):
    V = []
    for W, b in weights:    
        V.append([_np.zeros_like(W.value), _np.zeros_like(b.value)])
    def optimizer():
        """Stochastic gradient descent with momentum."""
        for i, (W, b) in enumerate(weights):
            # Set momentum
            V[i][0] = beta * V[i][0] + (1 - beta) * W.grad
            V[i][1] = beta * V[i][1] + (1 - beta) * b.grad
            # Descend gradients
            W.value -= learning_rate * V[i][0]
            b.value -= learning_rate * V[i][1]
            # Zero grads
            W.grad, b.grad = None, None
            W.ref, b.ref = 0, 0
    return optimizer

# class sgd_momentum:
    
#     def __init__(self, weights, learning_rate=0.001, beta=0.9):
#         weights = weights
#         learning_rate = learning_rate
#         beta = beta
#         V = []
#         for W, b in weights:    
#             V.append([_np.zeros_like(W.value), _np.zeros_like(b.value)])

#     def run_optimizer(self):
#         """Stochastic gradient descent with momentum."""
#         num = 0
#         # print("before W:", weights[num][0])
#         # print("grad W:", weights[num][0].grad)
#         # print("before V:", V[num][0])
#         for i, (W, b) in enumerate(weights):
#             # if i == num:
#                 # print("should V:", beta * V[num][0] + (1 - beta) * W.grad)
#             # Set momentum
#             V[i][0] = beta * V[i][0] + (1 - beta) * W.grad
#             V[i][1] = beta * V[i][1] + (1 - beta) * b.grad
#             # Descend gradients
#             W.value -= learning_rate * V[i][0]
#             b.value -= learning_rate * V[i][1]
#             # Zero grads
#             W.grad, b.grad = None, None
#             # print("ref:", W.ref, b.ref)
#             W.ref, b.ref = 0, 0
#         # print("after V:", V[num][0])
#         # print("after W:", weights[num][0])
        