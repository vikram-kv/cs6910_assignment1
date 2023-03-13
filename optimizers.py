import numpy as np

# all optimizers must have functions "forward", "backward", "update_parameters". 
class Optimizer:
    def forward(self):
        pass
    def backward(self):
        pass
    def update_parameters(self):
        pass

# a helper function to generate list of zeroed out np matrices for optimizers. uses a neural network's arch to build these lists.
# used for storing momentum terms, and weighed sum of squared gradients etc ...
def get_lists(nn):
    weight_list, bias_list = [None for i in range(nn.hlayercount+2)], [None for i in range(nn.hlayercount+2)]
    for idx in range(1, nn.hlayercount + 1):
        weight_list[idx] = np.zeros((nn.hidden_sizes[idx], nn.hidden_sizes[idx - 1]))
        bias_list[idx] = np.zeros(nn.hidden_sizes[idx])
    # for output layer
    outidx = nn.hlayercount + 1
    weight_list[outidx] = np.zeros((nn.out_layer_size, nn.hidden_sizes[outidx - 1]))
    bias_list[outidx] = np.zeros(nn.out_layer_size)
    return weight_list, bias_list

# SGD Optimizer
class sgd(Optimizer):
    def __init__(self, nn, args):
        self.nn = nn
        self.forward, self.backward = nn.forward, nn.backward

    def update_parameters(self, weights, biases, learning_rate, batch_weight_gradient, batch_bias_gradient):
        eta = learning_rate
        for idx in range(1, self.nn.hlayercount + 2):
            weights[idx] -= eta * batch_weight_gradient[idx]
            biases[idx] -= eta * batch_bias_gradient[idx]

# Momentum Optimizer
class momentum(Optimizer):
    def __init__(self, nn, args):
        self.beta = args.momentum
        self.nn = nn
        self.weight_momentums, self.bias_momentums = get_lists(nn)
        # no change to weights at which gradient is found; so, normal nn forward and backward are used
        self.forward, self.backward = nn.forward, nn.backward
    
    def update_parameters(self, weights, biases, learning_rate, batch_weight_gradient, batch_bias_gradient):
        eta, beta = learning_rate, self.beta
        # momentum update rule
        for idx in range(1, self.nn.hlayercount + 2):
            self.weight_momentums[idx] = beta * self.weight_momentums[idx] + eta * batch_weight_gradient[idx]
            self.bias_momentums[idx] = beta * self.bias_momentums[idx] + eta * batch_bias_gradient[idx]
            weights[idx] -= self.weight_momentums[idx]
            biases[idx] -= self.bias_momentums[idx]

# Nesterov Accelerated Momentum Optimizer
class nag(Optimizer):
    def __init__(self, nn, args):
        self.beta = args.momentum
        self.nn = nn
        self.weight_momentums, self.bias_momentums = get_lists(nn)
    
    # function to compute partially updated weights using momentum history/nesterov trick
    def get_partial_update_parameters(self, weights, biases):
        beta = self.beta
        partial_updated_weights, partial_updated_biases = dict(), dict()
        for idx in range(1, self.nn.hlayercount + 2):
            partial_updated_weights[idx] = weights[idx] - beta * self.weight_momentums[idx]
            partial_updated_biases[idx] = biases[idx] - beta * self.bias_momentums[idx]
        return partial_updated_weights, partial_updated_biases

    # compute partial updated weights and then use nn forward with these weights
    def forward(self, weights, biases, input : np.array, true_label):
        partial_updated_weights, partial_updated_biases = self.get_partial_update_parameters(weights, biases)
        return self.nn.forward(partial_updated_weights, partial_updated_biases, input, true_label)

    # compute partial updated weights and then use nn backward with these weights
    def backward(self, weights, biases, true_label, outvalues, outderivs, fin_act_values):
        partial_updated_weights, partial_updated_biases = self.get_partial_update_parameters(weights, biases)
        return self.nn.backward(partial_updated_weights, partial_updated_biases, true_label, outvalues, outderivs, fin_act_values)

    def update_parameters(self, weights, biases, learning_rate, batch_weight_gradient, batch_bias_gradient):
        eta = learning_rate
        beta = self.beta

        for idx in range(1, self.nn.hlayercount + 2):
            self.weight_momentums[idx] = beta * self.weight_momentums[idx] + eta * batch_weight_gradient[idx]
            self.bias_momentums[idx] = beta * self.bias_momentums[idx] + eta * batch_bias_gradient[idx]
            weights[idx] -= self.weight_momentums[idx]
            biases[idx] -= self.bias_momentums[idx]

# RMSProp Optimizer
class rmsprop(Optimizer):
    def __init__(self, nn, args):
        self.beta, self.epsilon = args.beta, args.epsilon
        self.nn = nn
        self.discounted_squared_weights, self.discounted_squared_biases = get_lists(nn)
        self.forward, self.backward = nn.forward, nn.backward

    def update_parameters(self, weights, biases, learning_rate, batch_weight_gradient, batch_bias_gradient):
        eta, beta, eps = learning_rate, self.beta, self.epsilon
        for idx in range(1, self.nn.hlayercount + 2):
            self.discounted_squared_weights[idx] = beta * self.discounted_squared_weights[idx] + (1-beta) * np.square(batch_weight_gradient[idx])
            self.discounted_squared_biases[idx] = beta * self.discounted_squared_biases[idx] + (1-beta) * np.square(batch_bias_gradient[idx])
            weights[idx] -= batch_weight_gradient[idx] * eta / (eps + np.sqrt(self.discounted_squared_weights[idx]))
            biases[idx] -= batch_bias_gradient[idx] * eta / (eps + np.sqrt(self.discounted_squared_biases[idx]))

# ADAM Optimizer
class adam(Optimizer):
    def __init__(self, nn, args):
        self.beta1, self.beta2, self.epsilon = args.beta1, args.beta2, args.epsilon
        self.nn = nn
        self.weight_momentums, self.bias_momentums = get_lists(nn)
        self.discounted_squared_weights, self.discounted_squared_biases = get_lists(nn)
        self.forward, self.backward = nn.forward, nn.backward
        self.step_no = 0

    def update_parameters(self, weights, biases, learning_rate, batch_weight_gradient, batch_bias_gradient):
        eta, beta1, beta2, eps = learning_rate, self.beta1, self.beta2, self.epsilon
        self.step_no += 1
        t = self.step_no
        for idx in range(1, self.nn.hlayercount + 2):
            # adam momentum and squared gradient history computation
            self.weight_momentums[idx] = beta1 * self.weight_momentums[idx] + (1-beta1) * batch_weight_gradient[idx]
            self.bias_momentums[idx] = beta1 * self.bias_momentums[idx] + (1-beta1) * batch_bias_gradient[idx]
            self.discounted_squared_weights[idx] = beta2 * self.discounted_squared_weights[idx] + (1-beta2) * np.square(batch_weight_gradient[idx])
            self.discounted_squared_biases[idx] = beta2 * self.discounted_squared_biases[idx] + (1-beta2) * np.square(batch_bias_gradient[idx])

            # bias correction for numerator(momentum) and denominator(discounted squared gradient) factors
            weight_num_hat, bias_num_hat = self.weight_momentums[idx] / (1 - np.power(beta1, t)), self.bias_momentums[idx] / (1 - np.power(beta1, t))
            weight_den_hat, bias_den_hat = self.discounted_squared_weights[idx] / (1 - np.power(beta2, t)), self.discounted_squared_biases[idx] / (1 - np.power(beta2, t))

            # adam update
            weights[idx] -= weight_num_hat * eta / (eps + np.sqrt(weight_den_hat))
            biases[idx] -= bias_num_hat * eta / (eps + np.sqrt(bias_den_hat))

# Nesterov ADAM Optimizer
class nadam(Optimizer):
    def __init__(self, nn, args):
        self.beta1, self.beta2, self.epsilon = args.beta1, args.beta2, args.epsilon
        self.nn = nn
        self.weight_momentums, self.bias_momentums = get_lists(nn)
        self.discounted_squared_weights, self.discounted_squared_biases = get_lists(nn)
        self.forward, self.backward = nn.forward, nn.backward
        self.step_no = 0

    def update_parameters(self, weights, biases, learning_rate, batch_weight_gradient, batch_bias_gradient):
        eta, beta1, beta2, eps = learning_rate, self.beta1, self.beta2, self.epsilon
        self.step_no += 1
        t = self.step_no
        for idx in range(1, self.nn.hlayercount + 2):
            # nadam momentum and squared gradient history computation
            self.weight_momentums[idx] = beta1 * self.weight_momentums[idx] + (1-beta1) * batch_weight_gradient[idx]
            self.bias_momentums[idx] = beta1 * self.bias_momentums[idx] + (1-beta1) * batch_bias_gradient[idx]
            self.discounted_squared_weights[idx] = beta2 * self.discounted_squared_weights[idx] + (1-beta2) * np.square(batch_weight_gradient[idx])
            self.discounted_squared_biases[idx] = beta2 * self.discounted_squared_biases[idx] + (1-beta2) * np.square(batch_bias_gradient[idx])

            # bias correction for numerator(momentum) and denominator(discounted squared gradient) factors
            weight_num_hat, bias_num_hat = self.weight_momentums[idx] / (1 - np.power(beta1, t)), self.bias_momentums[idx] / (1 - np.power(beta1, t))
            weight_den_hat, bias_den_hat = self.discounted_squared_weights[idx] / (1 - np.power(beta2, t)), self.discounted_squared_biases[idx] / (1 - np.power(beta2, t))

            # nadam update
            weights[idx] -= (beta1 * weight_num_hat + (1-beta1) * batch_weight_gradient[idx]/(1 - np.power(beta1, t))) * eta / (eps + np.sqrt(weight_den_hat))
            biases[idx] -= (beta1 * bias_num_hat + beta1 + (1-beta1) * batch_bias_gradient[idx]/(1 - np.power(beta1, t))) * eta / (eps + np.sqrt(bias_den_hat))