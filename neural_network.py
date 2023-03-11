import numpy as np
from activation_functions import *
from loss_functions import *
from optimizers import *
from tqdm import tqdm

class NeuralNetwork:

    def __init__(self, args, num_classes, in_dim):
        self.hlayercount = args.num_layers
        # hidden layers are numbered from 1 ... hlayercount; outputlayer is hlayercount + 1
        self.hidden_actfn = [get_act_func_and_deriv(args.activation) for _ in range(self.hlayercount + 1)]
        self.hidden_sizes = [args.hidden_size for _ in range(self.hlayercount + 2)]
        self.hidden_sizes[0] = in_dim # for ease of weight init
        self.hidden_sizes[self.hlayercount + 1] = num_classes # for ease of weight init
        self.init_method = args.weight_init
        self.weight_decay = args.weight_decay
        self.loss_fn = args.loss
        self.out_layer_size = num_classes
        self.in_layer_size = in_dim
        self.optimizer = self.get_optimizer_by_name(args)
        self.loss_fn = get_loss_by_name(args.loss)
        self.act_name = args.activation
    
    # add custom optimizer's entry here
    def get_optimizer_by_name(self, args):
        if (args.optimizer == 'sgd'):
            return sgd(self, args)
        elif (args.optimizer == 'momentum'):
            return momentum(self, args)
        elif (args.optimizer == 'nag'):
            return nag(self, args)
        elif (args.optimizer == 'rmsprop'):
            return rmsprop(self, args)
        elif (args.optimizer == 'adam'):
            return adam(self, args)
        elif (args.optimizer == 'nadam'):
            return nadam(self, args)
        else:
            Exception('Optimizer Not Implemented'); exit(-1)

    def init_parameters(self):
        weights, biases = [None for i in range(self.hlayercount+2)], [None for i in range(self.hlayercount+2)]
        np.random.seed(2)
        # gains for xavier init [scaling factors for better performance]
        gains = {'tanh' : 5/3, 'relu' : np.sqrt(2), 'leakyrelu' : np.sqrt(2), 'elu':np.sqrt(2)}
        gain = 1.0
        if self.act_name in gains.keys():
            gain = gains[self.act_name]
        if self.init_method == 'xavier':
            # uniform dist Xavier initialization
            for idx in range(1, self.hlayercount+2):
                ran = np.sqrt(6/(self.hidden_sizes[idx] + self.hidden_sizes[idx-1]))
                weights[idx] = gain * np.random.uniform(low=-ran,high=ran,size=(self.hidden_sizes[idx], self.hidden_sizes[idx - 1]))
                biases[idx] = gain * np.random.uniform(low=-ran,high=ran,size=(self.hidden_sizes[idx]))
        
        elif self.init_method == 'he':
            # uniform kaiming (He) initialization with mode = fanin
            for idx in range(1, self.hlayercount+2):
                ran = np.sqrt(3/(self.hidden_sizes[idx-1]))
                weights[idx] = gain * np.random.uniform(low=-ran,high=ran,size=(self.hidden_sizes[idx], self.hidden_sizes[idx - 1]))
                biases[idx] = gain * np.random.uniform(low=-ran,high=ran,size=(self.hidden_sizes[idx]))
        
        else:
            # random initialization with stdev of 0.1
            gain = 0.1
            for idx in range(1, self.hlayercount+2):
                weights[idx] = gain * np.random.randn(self.hidden_sizes[idx], self.hidden_sizes[idx - 1])
                biases[idx] = gain * np.random.randn(self.hidden_sizes[idx])

        return weights, biases

    # input is of shape -> (in_dim, batchsize); labels is a list of labels of size batchsize
    def forward(self, weights, biases, input, labels):
        outvalues, outderivs = [np.copy(input)], [None]
        for idx in range(1, self.hlayercount + 1):
            next_act_values = ((weights[idx] @ input).T + biases[idx]).T
            next_out_values = self.hidden_actfn[idx][0](next_act_values)
            next_out_derivs = self.hidden_actfn[idx][1](next_out_values)
            outvalues.append(next_out_values)
            outderivs.append(next_out_derivs)
            input = next_out_values
        
        # final layer - get act values and compute loss
        finidx = self.hlayercount + 1
        fin_act_values = ((weights[finidx] @ input).T + biases[finidx]).T
        Loss = self.loss_fn
        y_pred, loss = Loss.compute_loss_and_final_layer_gradients_preact(fin_act_values, labels, False)
        return outvalues, outderivs, fin_act_values, y_pred, loss

    # fin_act_values -> final_layer_act_values (shape = (out_dim, batchsize)); outvalues and outderivs are returned by
    # forward -> to save computation; labels -> list of labels of size batchsize
    def backward(self, weights, biases, labels, outvalues, outderivs, fin_act_values):
        weight_gradients, bias_gradients = [None for i in range(self.hlayercount+2)], [None for i in range(self.hlayercount+2)] # to be returned

        Loss = self.loss_fn
        _, _, gradients_act_values = Loss.compute_loss_and_final_layer_gradients_preact(fin_act_values, labels, True)
        weight_gradients[self.hlayercount + 1]  = np.einsum('ac,bc->abc', gradients_act_values, outvalues[self.hlayercount])
        bias_gradients[self.hlayercount + 1] = np.copy(gradients_act_values)

        for idx in range(self.hlayercount, 0, -1):
            loss_grad_outputs = np.transpose(weights[idx + 1]) @ (gradients_act_values)
            gradients_act_values = np.multiply(loss_grad_outputs, outderivs[idx])
            weight_gradients[idx]  = np.einsum('ac,bc->abc', gradients_act_values, outvalues[idx-1])
            bias_gradients[idx] = np.copy(gradients_act_values)
        
        return weight_gradients, bias_gradients

    # function to split X and y into batches of size batch_size. Required to exploit
    # power of numpy on matrix operations by "forwarding" and "backwarding" all examples
    # in a batch simultaneously.
    def make_batches(self, X, y, batch_size):
        batch_X, batch_y = [], []
        for i in range(0, len(X), batch_size):
            if (i + batch_size < len(X)):
                next_X = X[i:(i + batch_size)].T
                next_y = y[i:(i + batch_size)]
            elif (i < len(X)):
                next_X = X[i:].T
                next_y = y[i:]
            else:
                break
            batch_X.append(next_X)
            batch_y.append(next_y)
        return batch_X, batch_y
    
    # function to test the current parameters of the network against validation/test data
    # again, these data must be in batches
    def test(self, weights, biases, val_batches):
        total_count = 0
        total_correct = 0
        total_loss = 0.0
        for X, y in zip(val_batches[0], val_batches[1]):
            _, _, _, y_pred, loss = self.optimizer.forward(weights, biases, X, y)
            total_loss += loss
            pred_labels = np.argmax(y_pred, axis=0)
            total_count += len(y)
            total_correct += len(np.where(pred_labels == y)[0])
        
        test_acc, test_loss = total_correct/total_count, total_loss/total_count
        print(f'accuracy = {test_acc}; loss = {test_loss}')
        return test_acc, test_loss

    # function to train the neural network using train_data and perform validation
    # (for hyperparameter fine-tuning) using val_data
    def train(self, train_data, val_data, epochs, batchsize, learning_rate):
        # batchify the data
        (train_X, train_y) = train_data
        (val_X, val_y) = val_data
        train_batches = self.make_batches(train_X, train_y, batchsize)
        val_batches = self.make_batches(val_X, val_y, batchsize)

        weights, biases = self.init_parameters()
        for e in range(epochs):
            print(f'EPOCH - {e+1}')
            batch_count = 0
            # forward and backward over all data (1 epoch)
            train_loss = 0
            agg_crct = 0
            for X, y in tqdm(zip(train_batches[0], train_batches[1]), total=len(train_batches[0])):
                batch_weight_gradient, batch_bias_gradient = [None for i in range(self.hlayercount+2)], [None for i in range(self.hlayercount+2)]
                outvalues, outderivs, fin_act_values, y_pred, loss = self.optimizer.forward(weights, biases, X, y)
                pred_labels = np.argmax(y_pred, axis=0)
                weight_gradients, bias_gradients = self.optimizer.backward(weights, biases, y, outvalues, outderivs, fin_act_values)

                train_loss += loss
                agg_crct += len(np.where(pred_labels == y)[0])

                for idx in range(1, self.hlayercount+2):
                    batch_bias_gradient[idx] = np.sum(bias_gradients[idx], axis=-1)
                    batch_weight_gradient[idx] = np.sum(weight_gradients[idx], axis=-1)
                
                # make log and test after every batch / epoch
                self.optimizer.update_parameters(weights, biases, learning_rate, batch_weight_gradient, batch_bias_gradient)
                batch_count += 1
                if (batch_count % 100 == 0):
                    _, _ = self.test(weights, biases, val_batches)

        _, _ = self.test(weights, biases, val_batches)
