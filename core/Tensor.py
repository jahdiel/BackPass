import numpy as np

class Tensor:

    def __init__(self, value, func=None, parents=None, grad_func=None, want_grad=False):

        self.value = value
        self.func = func
        self.grad_func = grad_func
        self.parents = parents
        self.grad = None
        self.want_grad = want_grad
        self.ref = 0
        self.set_size()
    
    def set_size(self):
        if isinstance(self.value, np.ndarray):
            self.size = self.value.size
        elif isinstance(self.value, list):
            self.size = len(self.value)
        else:
            return 1

    def pass_grads(self):
        # Calculate VJP
        val_vjp = self.grad_func(*self.parents, self.grad)
        # Add grads to arguments
        for i in range(len(self.parents)):
            if self.parents[i].grad == None: self.parents[i].grad = 0
            self.parents[i].grad += val_vjp[i]

    def __repr__(self):
        if self.parents == None:
            return '<backpass.core.Tensor.Tensor {} -leaf ref: {}>'.format(self.value, self.ref)
        return '<backpass.core.Tensor.Tensor {} ref: {}>'.format(self.value, self.ref)