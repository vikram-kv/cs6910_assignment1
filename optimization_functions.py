import numpy as np

# all optimizers must implement this class
class Optimizer:
    def __init__(self):
        pass

    def update_parameters(self):
        pass

class sgd(Optimizer):
    def __init__(self, nn, args):
        self.hlayercount = nn.hlayercount
        self.update_parameters = self.sgd_update
        self.forward = nn.forward
        self.backward = nn.backward

    def sgd_update(self, weights, biases, learning_rate, agg_weight_changes, agg_bias_changes):
        eta = learning_rate
        for idx in range(1, self.hlayercount + 2):
            weights[idx] -= eta * agg_weight_changes[idx]
            biases[idx] -= eta * agg_bias_changes[idx]

class momentum(Optimizer):
    def __init__(self, nn, args):
        self.beta = args.mom

        self.weight_momentums, self.bias_momentums = dict(), dict()
        for idx in range(1, nn.hlayercount + 1):
            self.weight_momentums[idx] = np.zeros((nn.hidden_sizes[idx], nn.hidden_sizes[idx - 1]))
            self.bias_momentums[idx] = np.zeros(nn.hidden_sizes[idx])
        # for output layer
        outidx = nn.hlayercount + 1
        self.weight_momentums[outidx] = np.zeros((nn.out_layer_size, nn.hidden_sizes[outidx - 1]))
        self.bias_momentums[outidx] = np.zeros(nn.out_layer_size)

        self.hlayercount = nn.hlayercount
        self.update_parameters = self.mom_update
        self.forward = nn.forward
        self.backward = nn.backward
    
    def mom_update(self, weights, biases, learning_rate, agg_weight_changes, agg_bias_changes):
        eta = learning_rate
        beta = self.beta

        for idx in range(1, self.hlayercount + 2):
            self.weight_momentums[idx] = beta * self.weight_momentums[idx] + eta * agg_weight_changes[idx]
            self.bias_momentums[idx] = beta * self.bias_momentums[idx] + eta * agg_bias_changes[idx]
            weights[idx] -= self.weight_momentums[idx]
            biases[idx] -= self.bias_momentums[idx]

class nag(Optimizer):
    def __init__(self, nn, args):
        self.beta = args.mom
        self.nn = nn

        self.weight_momentums, self.bias_momentums = dict(), dict()
        for idx in range(1, nn.hlayercount + 1):
            self.weight_momentums[idx] = np.zeros((nn.hidden_sizes[idx], nn.hidden_sizes[idx - 1]))
            self.bias_momentums[idx] = np.zeros(nn.hidden_sizes[idx])
        # for output layer
        outidx = nn.hlayercount + 1
        self.weight_momentums[outidx] = np.zeros((nn.out_layer_size, nn.hidden_sizes[outidx - 1]))
        self.bias_momentums[outidx] = np.zeros(nn.out_layer_size)
        self.hlayercount = nn.hlayercount
        self.update_parameters = self.nag_update
    
    def get_partial_update_parameters(self, weights, biases):
        beta = self.beta
        partial_updated_weights, partial_updated_biases = dict(), dict()
        for idx in range(1, self.hlayercount + 2):
            partial_updated_weights[idx] = weights[idx] - beta * self.weight_momentums[idx]
            partial_updated_biases[idx] = biases[idx] - beta * self.bias_momentums[idx]
        return partial_updated_weights, partial_updated_biases

    def forward(self, weights, biases, input : np.array, true_label):
        partial_updated_weights, partial_updated_biases = self.get_partial_update_parameters(weights, biases)
        return self.nn.forward(partial_updated_weights, partial_updated_biases, input, true_label)

    def backward(self, weights, biases, true_label, outvalues, outderivs):
        partial_updated_weights, partial_updated_biases = self.get_partial_update_parameters(weights, biases)
        return self.nn.backward(partial_updated_weights, partial_updated_biases, true_label, outvalues, outderivs)

    def nag_update(self, weights, biases, learning_rate, agg_weight_changes, agg_bias_changes):
        eta = learning_rate
        beta = self.beta

        for idx in range(1, self.hlayercount + 2):
            self.weight_momentums[idx] = beta * self.weight_momentums[idx] + eta * agg_weight_changes[idx]
            self.bias_momentums[idx] = beta * self.bias_momentums[idx] + eta * agg_bias_changes[idx]
            weights[idx] -= self.weight_momentums[idx]
            biases[idx] -= self.bias_momentums[idx]
    