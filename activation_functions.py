import numpy as np

# sigmoid activation function and derivative
def sigmoid_value(x):
    return 1.0/(1.0 + np.exp(-x))

def sigmoid_deriv(fx):
    return (1.0 - fx) * fx

# relu activation function and derivative
def relu_value(x):
    return (0.0 if x < 0 else x)

def relu_deriv(fx):
    return (0.0 if fx < 0 else 1.0)

# tanh activation function and derivative
def tanh_value(x):
    y, z = np.exp(x), np.exp(-x)
    return (y-z)/(y+z)

def tanh_deriv(fx):
    return (1.0 - fx) * (1.0 + fx)

# linear activation function and derivative
def linear_value(x):
    return x

def linear_deriv(fx):
    return 1.0

# function to return the function operator and derivative operator for a activation function by name
def get_act_func_and_deriv(name : str):
    if (name == 'linear'):
        return linear_value, linear_deriv
    elif (name == 'tanh'):
        return tanh_value, tanh_deriv
    elif (name == 'sigmoid'):
        return sigmoid_value, sigmoid_deriv
    elif (name == 'relu'):
        return relu_value, relu_deriv
    else:
        raise Exception('Activation Function Not Implemented'); exit(-1)