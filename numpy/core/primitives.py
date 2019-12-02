import numpy as np
from backpass.core.wrap_primitives import primitive

@primitive
def sum(a):
    return np.sum(a)

@primitive
def square(a):
    return np.square(a)

@primitive
def log(a):
    return np.log(a)

@primitive
def log2(a):
    return np.log2(a)

@primitive
def log10(a):
    return np.log10(a)

@primitive
def add(a, b):
    return np.add(a, b)

@primitive
def multiply(a, b):
    return np.multiply(a, b)

@primitive
def subtract(a, b):
    return np.subtract(a, b)

@primitive
def dot(a, b):
    return np.dot(a, b)

