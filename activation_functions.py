import numpy as np

# sigmoid activation function and derivative
def sigmoid_value(x):
    return 1.0/(1.0 + np.exp(-x))

def sigmoid_deriv(x):
    fx = sigmoid_value(x)
    return (1.0 - fx) * fx

# relu activation function and derivative
def relu_value(x):
    return (0.0 if x < 0 else x)

def relu_deriv(x):
    return (0.0 if x < 0 else 1.0)

# tanh activation function and derivative
def tanh_value(x):
    y, z = np.exp(x), np.exp(-x)
    return (y-z)/(y+z)

def tanh_deriv(x):
    fx = tanh_value(x)
    return (1.0 - fx) * (1.0 + fx)

# linear activation function and derivative
def linear_value(x):
    return x

def linear_deriv(x):
    return 1.0