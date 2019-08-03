
class Tensor:

    def __init__(self, value, func, parents):

        self.value = value
        self.func = func
        self.parents = parents

    def __repr__(self):
        return '<backpass.core.Tensor.Tensor {} >'.format(self.value)