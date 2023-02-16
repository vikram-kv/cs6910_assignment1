from helper_functions import *
import numpy as np

class NeuralNetwork:
    def __init__(self, args, num_classes, in_dim):
        self.hlayercount = args.num_hidden_layers
        self.act_fn = [args.act_func for _ in range(self.hlayercount+1)]
        self.hidden_sizes = [args.hidden_size for _ in range(self.hlayercount+1)]
        self.init_method = args.w_i
        self.weight_decay = args.w_d
        self.loss_fn = args.loss
        self.out_layer_size = num_classes
        self.in_layer_size = in_dim
        self.hidden_sizes[0] = in_dim # for ease later
        self.create_dicts()
        self.init_parameters()
        pass
    
    def create_dicts(self):
        self.weights = dict()
        self.biases = dict()
        self.act_values = dict()
        self.out_values = dict()
        self.out_derivs = dict()
        self.loss_grad_act = dict()
        self.loss_grad_outputs = dict()
        self.weight_changes = dict()
        self.biases_changes = dict()

    # implement Xavier initialization
    def init_parameters(self):
        if self.init_method == 'Xavier':
            pass
        else:
            for idx in range(1, self.hlayercount + 1):
                self.weights[idx] = np.random.randn(self.hidden_sizes[idx], self.hidden_sizes[idx - 1])
                self.biases[idx] = np.random.randn(self.hidden_sizes[idx])
            
            # for output layer
            outidx = self.hlayercount + 1
            self.weights[outidx] = np.random.randn(self.hidden_sizes[outidx], self.hidden_sizes[outidx - 1])
            self.biases[outidx] = np.random.randn(self.hidden_sizes[outidx])

    def forward(self, input : np.array, true_label):
        incopy = np.copy(input)
        self.out_values[0] = incopy
        for idx in range(1, self.hlayercount + 1):
            l, m, n = forward_one_layer(self.weights, self.biases, idx, incopy, self.act_fn[idx])
            self.act_values[idx], self.out_values[idx], self.out_derivs[idx] = l, m, n
            incopy = np.copy(m)
        
        # final layer - get act values and compute loss
        l, _, _ = forward_one_layer(self.weights, self.biases, self.hlayercount + 1, incopy)
        self.act_values[self.hlayercount + 1] = l
        m = self.out_values[self.hlayercount + 1] = safe_softmax(l)
        self.loss = calculate_loss(self.loss_fn, m, true_label)

    def backward(self, true_label):
        self.loss_grad_act[self.hlayercount + 1] = loss_grad_fl_layer_act_values(self.loss_fn, true_label, self.out_values[self.hlayercount + 1])
        for idx in range(self.hlayercount, 0, -1):
            self.loss_grad_outputs[idx] = loss_grad_hd_layer_output_values(idx, self.weights, self.loss_grad_act[idx+1])
            self.loss_grad_act[idx] = loss_grad_hd_layer_act_values(self.loss_grad_outputs[idx], self.out_derivs[idx])
            self.weight_changes[idx], self.biases_changes[idx] = compute_parameter_derivatives(self.loss_grad_act[idx], self.out_values[idx-1])

    # need to include optimizers here onwards
    def update_parameters(self, learning_rate, agg_weight_changes, agg_bias_changes):
        eta = learning_rate
        for idx in range(1, self.hlayercount + 2):
            self.weights[idx] = - eta * agg_weight_changes[idx]
            self.biases[idx] = - eta * agg_bias_changes[idx]

    def refresh_aggregates(self):
        agg_weight_changes = dict()
        agg_biases_changes = dict()
        agg_loss = 0.0
        agg_correct = 0
        for idx in self.weights:
            agg_weight_changes[idx] = np.zeros(self.weights[idx])
            agg_biases_changes[idx] = np.zeros(self.biases[idx])
        return agg_weight_changes, agg_biases_changes, agg_loss, agg_correct

    def test(self, val_data):
        val_X, val_y = val_data
        total_count = val_y.shape[0]
        total_correct = 0
        total_loss = 0.0

        for X, y in zip(val_X, val_y):
            self.forward(X, y)
            total_loss += self.loss
            y_pred = np.amax(self.out_values[self.hlayercount + 1])
            if (y == y_pred):
                total_correct += 1
        acc = total_correct / total_count
        print(f'accuracy = {acc}; loss = {total_loss}')
        return acc, total_loss
            
    # NOTE - ensure batch size divides total train data size
    def train(self, train_data, val_data, epochs, batchsize, learning_rate):
        train_X, train_y = train_data
        for i in range(epochs):
            num_samples = 0
            agg_weight_changes, agg_biases_changes, agg_loss, agg_crct = self.refresh_aggregates()
            for X, y in zip(train_X, train_y):
                self.forward(X, y)
                y_pred = np.amax(self.out_values[self.hlayercount + 1])
                self.backward(y)
                num_samples += 1
                agg_loss += self.loss
                if (y_pred == y):
                    agg_crct += 1

                for idx in self.weight_changes:
                    agg_weight_changes[idx] += self.weight_changes[idx]
                    agg_biases_changes[idx] += self.biases_changes[idx]

                if (num_samples % batchsize == 0):
                    # make log
                    self.update_parameters(learning_rate, agg_weight_changes, agg_biases_changes)
                    num_samples = 0
                    agg_weight_changes, agg_biases_changes, agg_loss, agg_crct = self.refresh_aggregates()
                    _, _ = self.test(val_data)

