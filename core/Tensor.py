
class Tensor:

    def __init__(self, value, func=None, parents=None, grad_func=None, kwargs=None, want_grad=False):

        self.value = value
        self.func = func
        self.grad_func = grad_func
        self.parents = parents
        self.kwargs = kwargs
        self.grad = None
        self.want_grad = want_grad
        self.ref = 0
        self.set_size()

    def pass_grads(self):
        # Calculate VJP
        parent_vals = [p.value if isinstance(p, Tensor) else p for p in self.parents]
        val_vjp = self.grad_func(*parent_vals, ans=self.value, grad=self.grad, **self.kwargs)
        # print(self.value, self.grad, self.grad_func, self.ref, val_vjp, "\n\n")
        # print(self.value, self.grad, self.grad_func, self.ref, "\n\n")
        # Add grads to arguments
        for i in range(len(self.parents)):
            if hasattr(self.parents[i], 'grad'):
                if self.parents[i].grad is None: self.parents[i].grad = val_vjp[i]
                else: self.parents[i].grad += val_vjp[i]
    
    def tensor_parents(self):
        return [parent for parent in self.parents if isinstance(parent, Tensor)]
    
    def set_size(self):
        if hasattr(self.value, "size"):
            self.size = self.value.size
        elif hasattr(self.value, "__len__"):
            self.size = len(self.value)
        else:
            self.size = 1
    
    def __getitem__(self, key):
        if hasattr(self.value, '__getitem__'):
            return self.value[key]
        raise TypeError("'Tensor' object is not subscriptable")

    def __repr__(self):
        if self.parents == None:
            return '<backpass.core.Tensor.Tensor {} -leaf ref: {}>'.format(self.value, self.ref)
        return '<backpass.core.Tensor.Tensor {} ref: {}>'.format(self.value, self.ref)

    def __neg__(self):
        import backpass.numpy as primitives
        return primitives.negative(self)

    def __eq__(self, other):
        import backpass.numpy as primitives
        return primitives.equal(self, other)

    def __add__(self, other):
        import backpass.numpy as primitives
        return primitives.add(self, other)

    def __radd__(self, other):
        import backpass.numpy as primitives
        return primitives.add(other, self)

    def __mul__(self, other):
        import backpass.numpy as primitives
        return primitives.multiply(self, other)
    
    def __rmul__(self, other):
        import backpass.numpy as primitives
        return primitives.multiply(other, self)

    def __sub__(self, other):
        import backpass.numpy as primitives
        return primitives.subtract(self, other)

    def __rsub__(self, other):
        import backpass.numpy as primitives
        return primitives.subtract(other, self)

    def __div__(self, other):
        import backpass.numpy as primitives
        return primitives.divide(self, other)
    
    def __rdiv__(self, other):
        import backpass.numpy as primitives
        return primitives.divide(other, self)

    def __truediv__(self, other):
        import backpass.numpy as primitives
        return primitives.true_divide(self, other)
    
    def __rtruediv__(self, other):
        import backpass.numpy as primitives
        return primitives.true_divide(other, self)

    def __hash__(self): return id(self)
