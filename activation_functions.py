import numpy as np

# new activation functions to be added following the template seen here.

# sigmoid activation function and derivative
def sigmoid_value(x):
    return 1.0/(1.0 + np.exp(-x))

def sigmoid_deriv(fx, x):
    return (1.0 - fx) * fx

# relu activation function and derivative
def relu_value(x):
    return np.where(x>0, x, 0.0)
    
def relu_deriv(fx, x):
    return np.where(fx>0, 1.0, 0.0)

# tanh activation function and derivative
def tanh_value(x):
    y, z = np.exp(x), np.exp(-x)
    return (y-z)/(y+z)

def tanh_deriv(fx, x):
    return (1.0 - fx) * (1.0 + fx)

# linear activation function and derivative
def linear_value(x):
    return x

def linear_deriv(fx, x):
    return np.ones_like(fx)

# leaky_relu activation function and derivative
def leakyrelu_value(x):
    return np.where(x>0, x, 0.05 * x)

def leakyrelu_deriv(fx, x):
    return np.where(fx>0, 1, 0.05)

# elu activation function and derivative
def elu_value(x):
    return np.where(x>0, x, 1.0 * (np.exp(x) - 1))
    
def elu_deriv(fx, x):
    return np.where(fx>0, 1.0, fx + 1.0)

# the most recent swish-1(beta = 1) activation function and derivative
def swish_value(x):
    sig_x = 1.0/(1.0 + np.exp(-x))
    return x * sig_x

def swish_deriv(fx, x):
    sig_x = 1.0/(1.0 + np.exp(-x))
    return fx + sig_x * (1- fx)

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
    elif (name == 'leakyrelu'):
        return leakyrelu_value, leakyrelu_deriv
    elif (name == 'elu'):
        return elu_value, elu_deriv
    elif (name == 'swish'):
        return swish_value, swish_deriv
    else:
        raise Exception('Activation Function Not Implemented'); exit(-1)