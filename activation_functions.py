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

### FORWARD PASS HELPERS ###

# performs the softmax activation step to get the final predicted posterior probabilities
def safe_softmax(input : np.array):
    prob = np.copy(input)
    prob -= np.max(prob)
    prob = np.exp(prob)
    prob = prob/(np.sum(prob))
    return prob

# apply the activation function to a vector of activation values of a hidden layer
# also returns the derivative of the act function at those activation values as a vector
def apply_act_fn_hidden_layer(act_values : np.array, act_fn : str):
    f, df = get_act_func_and_deriv(act_fn)
    out_values, out_derivs = np.copy(input), np.copy(input)
    for i in range(input.shape[0]):
        out_values[i] = f(act_values[i])
        out_derivs[i] = df(act_values[i])
    return out_values, out_derivs

# NOTE - thinking of 3d matrix. may need to switch to dictionary of matrices to accomodate diff no of neurons in each layer
# weights[i] -> matrix of weights of layer i (with jth row = weights of jth neuron), bias[i] -> bias for layer i (with jth entry
# as bias for neuron j)
def forward_one_layer(weights, bias, layeridx : int, input : int, act_fn = None):
    act_values = (weights[layeridx] @ input + bias[layeridx])
    outvalues, outderivs = None, None
    if act_fn != None:
        outvalues, outderivs  = apply_act_fn_hidden_layer(act_values, act_fn)
    return act_values, outvalues, outderivs

### BACKWARD PASS HELPERS ###

# returns the gradient of the loss function with respect to the output of the final layer
def loss_grad_fl_outputs(loss_fn : str, fl_output : str, true_lbl = None):
    ans = np.zeros(fl_output.shape)
    if (loss_fn == 'cross_entropy'):
        ans[true_lbl] = 1.0
        ans *= -1.0/(fl_output[true_lbl])
    elif (loss_fn == 'mean_squared_error'):
        sz = ans.shape[0]
        for i in range(sz):
            if (i == true_lbl):
                ans[i] = -(1 - fl_output[i])
            else:
                ans[i] = fl_output[i]
        ans *= 2.0 / sz
    else:
        raise Exception('Loss Function Not Implemented'); exit(-1)
    return ans

# NOTE - confirm once if we take softmax even for MSE loss
# returns the gradient of the loss wrt the activation values of the final layer
def loss_grad_fl_layer_act_values(loss_fn : str, true_lbl : int, fl_output : np.array):
    output_grad = loss_grad_fl_outputs(loss_fn, fl_output, true_lbl)
    ans = np.multiply(fl_output,output_grad)
    factor = np.dot(fl_output, output_grad)
    ans -= factor * (fl_output)
    return ans

# computes the loss gradient wrt layer outputs of HIDDEN layer with index = layeridx
def loss_grad_hd_layer_output_values(layeridx : int, weights, loss_grad_act_values_next_layer : np.array):
    return np.transpose(weights[layeridx + 1]) @ (loss_grad_act_values_next_layer)

# computes the loss gradient wrt act values of a hidden layer given the loss gradient wrt its output and 
# act function derivatives at the act values
def loss_grad_hd_layer_act_values(loss_grad_cur_layer_hd_output : np.array, cur_layer_act_fn_deriv : np.array):
    return np.multiply(loss_grad_cur_layer_hd_output, cur_layer_act_fn_deriv)
