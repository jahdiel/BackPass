import numpy as np
from backpass.core.wrap_primitives import primitive

@primitive
def sum(a):
    return np.sum(a)

@primitive
def square(a):
    return np.square(a)

@primitive
def add(a, b):
    return np.add(a, b)

@primitive
def multiply(a, b):
    return np.multiply(a, b)

@primitive
def subtract(a, b):
    return np.subtract(a, b)