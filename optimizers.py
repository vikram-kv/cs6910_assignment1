import numpy as np

# all optimizers must have functions "forward", "backward", "update_parameters". 
class Optimizer:
    def forward(self):
        pass
    def backward(self):
        pass
    def update_parameters(self):
        pass

class sgd(Optimizer):
    def __init__(self, nn, args):
        self.nn = nn
        self.update_parameters = self.update
        self.forward = nn.forward
        self.backward = nn.backward

    def update(self, weights, biases, learning_rate, agg_weight_gradients, agg_bias_gradients):
        eta = learning_rate
        for idx in range(1, self.nn.hlayercount + 2):
            weights[idx] = weights[idx] - eta * agg_weight_gradients[idx]
            biases[idx] -= eta * agg_bias_gradients[idx]

class momentum(Optimizer):
    def __init__(self, nn, args):
        self.beta = args.momentum
        self.nn = nn

        self.weight_momentums, self.bias_momentums = dict(), dict()
        for idx in range(1, nn.hlayercount + 1):
            self.weight_momentums[idx] = np.zeros((nn.hidden_sizes[idx], nn.hidden_sizes[idx - 1]))
            self.bias_momentums[idx] = np.zeros(nn.hidden_sizes[idx])
        # for output layer
        outidx = nn.hlayercount + 1
        self.weight_momentums[outidx] = np.zeros((nn.out_layer_size, nn.hidden_sizes[outidx - 1]))
        self.bias_momentums[outidx] = np.zeros(nn.out_layer_size)

        self.update_parameters = self.update
        self.forward = nn.forward
        self.backward = nn.backward
    
    def update(self, weights, biases, learning_rate, agg_weight_gradients, agg_bias_gradients):
        eta = learning_rate
        beta = self.beta

        for idx in range(1, self.nn.hlayercount + 2):
            self.weight_momentums[idx] = beta * self.weight_momentums[idx] + eta * agg_weight_gradients[idx]
            self.bias_momentums[idx] = beta * self.bias_momentums[idx] + eta * agg_bias_gradients[idx]
            weights[idx] -= self.weight_momentums[idx]
            biases[idx] -= self.bias_momentums[idx]

class nag(Optimizer):
    def __init__(self, nn, args):
        self.beta = args.momentum
        self.nn = nn

        self.weight_momentums, self.bias_momentums = dict(), dict()
        for idx in range(1, nn.hlayercount + 1):
            self.weight_momentums[idx] = np.zeros((nn.hidden_sizes[idx], nn.hidden_sizes[idx - 1]))
            self.bias_momentums[idx] = np.zeros(nn.hidden_sizes[idx])
        # for output layer
        outidx = nn.hlayercount + 1
        self.weight_momentums[outidx] = np.zeros((nn.out_layer_size, nn.hidden_sizes[outidx - 1]))
        self.bias_momentums[outidx] = np.zeros(nn.out_layer_size)
        self.update_parameters = self.update
    
    def get_partial_update_parameters(self, weights, biases):
        beta = self.beta
        partial_updated_weights, partial_updated_biases = dict(), dict()
        for idx in range(1, self.nn.hlayercount + 2):
            partial_updated_weights[idx] = weights[idx] - beta * self.weight_momentums[idx]
            partial_updated_biases[idx] = biases[idx] - beta * self.bias_momentums[idx]
        return partial_updated_weights, partial_updated_biases

    def forward(self, weights, biases, input : np.array, true_label):
        partial_updated_weights, partial_updated_biases = self.get_partial_update_parameters(weights, biases)
        return self.nn.forward(partial_updated_weights, partial_updated_biases, input, true_label)

    def backward(self, weights, biases, true_label, outvalues, outderivs):
        partial_updated_weights, partial_updated_biases = self.get_partial_update_parameters(weights, biases)
        return self.nn.backward(partial_updated_weights, partial_updated_biases, true_label, outvalues, outderivs)

    def update(self, weights, biases, learning_rate, agg_weight_gradients, agg_bias_gradients):
        eta = learning_rate
        beta = self.beta

        for idx in range(1, self.nn.hlayercount + 2):
            self.weight_momentums[idx] = beta * self.weight_momentums[idx] + eta * agg_weight_gradients[idx]
            self.bias_momentums[idx] = beta * self.bias_momentums[idx] + eta * agg_bias_gradients[idx]
            weights[idx] -= self.weight_momentums[idx]
            biases[idx] -= self.bias_momentums[idx]

class rmsprop(Optimizer):
    def __init__(self, nn, args):
        self.beta = args.beta
        self.nn = nn
        self.epsilon = args.epsilon

        self.weight_denoms, self.bias_denoms = dict(), dict()
        for idx in range(1, nn.hlayercount + 1):
            self.weight_denoms[idx] = np.zeros((nn.hidden_sizes[idx], nn.hidden_sizes[idx - 1]))
            self.bias_denoms[idx] = np.zeros(nn.hidden_sizes[idx])

        # for output layer
        outidx = nn.hlayercount + 1
        self.weight_denoms[outidx] = np.zeros((nn.out_layer_size, nn.hidden_sizes[outidx - 1]))
        self.bias_denoms[outidx] = np.zeros(nn.out_layer_size)
        self.forward = nn.forward
        self.backward = nn.backward
        self.update_parameters = self.update

    def update(self, weights, biases, learning_rate, agg_weight_gradients, agg_bias_gradients):
        eta = learning_rate
        beta = self.beta
        epsilon = self.epsilon

        for idx in range(1, self.nn.hlayercount + 2):
            self.weight_denoms[idx] = beta * self.weight_denoms[idx] + (1-beta) * np.square(agg_weight_gradients[idx])
            self.bias_denoms[idx] = beta * self.bias_denoms[idx] + (1-beta) * np.square(agg_bias_gradients[idx])
            weights[idx] -= agg_weight_gradients[idx] * (eta) / np.sqrt(epsilon + self.weight_denoms[idx])
            biases[idx] -= agg_bias_gradients[idx] * (eta) / np.sqrt(epsilon + self.bias_denoms[idx])

class adam(Optimizer):
    def __init__(self, nn, args):
        self.beta1 = args.beta1
        self.beta2 = args.beta2
        self.nn = nn
        self.epsilon = args.epsilon

        self.weight_denoms, self.bias_denoms = dict(), dict()
        for idx in range(1, nn.hlayercount + 1):
            self.weight_denoms[idx] = np.zeros((nn.hidden_sizes[idx], nn.hidden_sizes[idx - 1]))
            self.bias_denoms[idx] = np.zeros(nn.hidden_sizes[idx])

        # for output layer
        outidx = nn.hlayercount + 1
        self.weight_denoms[outidx] = np.zeros((nn.out_layer_size, nn.hidden_sizes[outidx - 1]))
        self.bias_denoms[outidx] = np.zeros(nn.out_layer_size)

        self.weight_momentums, self.bias_momentums = dict(), dict()
        for idx in range(1, nn.hlayercount + 1):
            self.weight_momentums[idx] = np.zeros((nn.hidden_sizes[idx], nn.hidden_sizes[idx - 1]))
            self.bias_momentums[idx] = np.zeros(nn.hidden_sizes[idx])

        # for output layer
        outidx = nn.hlayercount + 1
        self.weight_momentums[outidx] = np.zeros((nn.out_layer_size, nn.hidden_sizes[outidx - 1]))
        self.bias_momentums[outidx] = np.zeros(nn.out_layer_size)

        self.forward = nn.forward
        self.backward = nn.backward
        self.update_parameters = self.update
        self.step_no = 0

    def update(self, weights, biases, learning_rate, agg_weight_gradients, agg_bias_gradients):
        eta = learning_rate
        beta1 = self.beta1
        beta2 = self.beta2
        epsilon = self.epsilon

        self.step_no += 1
        t = self.step_no

        for idx in range(1, self.nn.hlayercount + 2):
            self.weight_momentums[idx] = beta1 * self.weight_momentums[idx] + (1-beta1) * agg_weight_gradients[idx]
            self.bias_momentums[idx] = beta1 * self.bias_momentums[idx] + (1-beta1) * agg_bias_gradients[idx]
            self.weight_denoms[idx] = beta2 * self.weight_denoms[idx] + (1-beta2) * np.square(agg_weight_gradients[idx])
            self.bias_denoms[idx] = beta2 * self.bias_denoms[idx] + (1-beta2) * np.square(agg_bias_gradients[idx])

            # normalize momentum and denominator factor
            cur_weight_momentum_hat, cur_bias_momentum_hat = self.weight_momentums[idx] / (1 - np.power(beta1, t)), self.bias_momentums[idx] / (1 - np.power(beta1, t))
            cur_weight_denom_hat, cur_bias_denom_hat = self.weight_denoms[idx] / (1 - np.power(beta2, t)), self.bias_denoms[idx] / (1 - np.power(beta2, t))

            weights[idx] -= cur_weight_momentum_hat * (eta) / np.sqrt(epsilon + cur_weight_denom_hat)
            biases[idx] -= cur_bias_momentum_hat * (eta) / np.sqrt(epsilon + cur_bias_denom_hat)

class nadam(Optimizer):
    def __init__(self, nn, args):
        self.beta1 = args.beta1
        self.beta2 = args.beta2
        self.nn = nn
        self.epsilon = args.epsilon

        self.weight_denoms, self.bias_denoms = dict(), dict()
        for idx in range(1, nn.hlayercount + 1):
            self.weight_denoms[idx] = np.zeros((nn.hidden_sizes[idx], nn.hidden_sizes[idx - 1]))
            self.bias_denoms[idx] = np.zeros(nn.hidden_sizes[idx])

        # for output layer
        outidx = nn.hlayercount + 1
        self.weight_denoms[outidx] = np.zeros((nn.out_layer_size, nn.hidden_sizes[outidx - 1]))
        self.bias_denoms[outidx] = np.zeros(nn.out_layer_size)

        self.weight_momentums, self.bias_momentums = dict(), dict()
        for idx in range(1, nn.hlayercount + 1):
            self.weight_momentums[idx] = np.zeros((nn.hidden_sizes[idx], nn.hidden_sizes[idx - 1]))
            self.bias_momentums[idx] = np.zeros(nn.hidden_sizes[idx])

        # for output layer
        outidx = nn.hlayercount + 1
        self.weight_momentums[outidx] = np.zeros((nn.out_layer_size, nn.hidden_sizes[outidx - 1]))
        self.bias_momentums[outidx] = np.zeros(nn.out_layer_size)

        self.forward = nn.forward
        self.backward = nn.backward
        self.update_parameters = self.update
        self.step_no = 0

    def update(self, weights, biases, learning_rate, agg_weight_gradients, agg_bias_gradients):
        eta = learning_rate
        beta1 = self.beta1
        beta2 = self.beta2
        epsilon = self.epsilon

        self.step_no += 1
        t = self.step_no

        for idx in range(1, self.nn.hlayercount + 2):
            self.weight_momentums[idx] = beta1 * self.weight_momentums[idx] + (1-beta1) * agg_weight_gradients[idx]
            self.bias_momentums[idx] = beta1 * self.bias_momentums[idx] + (1-beta1) * agg_bias_gradients[idx]
            self.weight_denoms[idx] = beta2 * self.weight_denoms[idx] + (1-beta2) * np.square(agg_weight_gradients[idx])
            self.bias_denoms[idx] = beta2 * self.bias_denoms[idx] + (1-beta2) * np.square(agg_bias_gradients[idx])

            cur_weight_momentum_hat, cur_bias_momentum_hat = self.weight_momentums[idx] / (1 - np.power(beta1, t)), self.bias_momentums[idx] / (1 - np.power(beta1, t))
            cur_weight_denom_hat, cur_bias_denom_hat = self.weight_denoms[idx] / (1 - np.power(beta2, t)), self.bias_denoms[idx] / (1 - np.power(beta2, t))

            # nesterov trick update
            weights[idx] -= (beta1 * cur_weight_momentum_hat + (1-beta1) * agg_weight_gradients[idx]) * (eta) / np.sqrt(epsilon + cur_weight_denom_hat)
            biases[idx] -= (beta1 * cur_bias_momentum_hat + (1-beta1) * agg_bias_gradients[idx]) * (eta) / np.sqrt(epsilon + cur_bias_denom_hat)
