# Code file for the neural network with the backprogation algorithm implementation
import numpy as np
from activation_functions import *
from loss_functions import *
from optimizers import *
from tqdm import tqdm
import wandb
from sklearn.metrics import confusion_matrix
import plotly.express as px

class NeuralNetwork:
    # init method receives all the arch parameters for the neural network and saves them in the object
    # saved details include num of hidden layers, their sizes, out_dim, in_dim, act func for every layer
    # loss object for the network, optimizer object for the network
    def __init__(self, args, num_classes, in_dim):
        self.hlayercount = args.num_hlayers
        # hidden layers are numbered from 1 ... hlayercount; outputlayer is hlayercount + 1
        self.hidden_actfn = [get_act_func_and_deriv(args.activation_function) for _ in range(self.hlayercount + 1)]
        self.hidden_sizes = [args.hidden_size for _ in range(self.hlayercount + 2)]
        self.hidden_sizes[0] = in_dim # for ease of weight init
        self.hidden_sizes[self.hlayercount + 1] = num_classes # for ease of weight init
        self.init_method = args.weight_init
        self.weight_decay = args.weight_decay
        self.out_layer_size = num_classes
        self.in_layer_size = in_dim
        self.optimizer = self.get_optimizer_by_name(args)
        self.loss_fn = get_loss_by_name(args.loss)
        self.act_name = args.activation_function
    
    # add custom optimizer's entry here. Optimizer is an object with the necessary gradient data
    # stored internally for every layer. For example, momentum opt stores weighted gradient sum
    # as a history measure internally for its functionality.
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

    # function to initialize all the weights and biases in the network. Supported init methods
    # include random, xavier and he
    def init_parameters(self):
        weights, biases = [None for i in range(self.hlayercount+2)], [None for i in range(self.hlayercount+2)]
        np.random.seed(42)
        # gains for xavier init [scaling factors] - did not improve performance; hence removed
        # gains = {'tanh' : 5/3, 'relu' : np.sqrt(2), 'leakyrelu' : np.sqrt(2), 'elu':np.sqrt(2)}
        # if self.act_name in gains.keys():
        #    gain = gains[self.act_name]
        gain = 1.0
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
            # random initialization with stddev of 0.15
            gain = 0.15
            for idx in range(1, self.hlayercount+2):
                weights[idx] = gain * np.random.randn(self.hidden_sizes[idx], self.hidden_sizes[idx - 1])
                biases[idx] = gain * np.random.randn(self.hidden_sizes[idx])

        return weights, biases

    # implements the forward pass for a batch
    # input is of shape -> (in_dim, batchsize); labels is a list of labels of size batchsize
    def forward(self, weights, biases, input, labels):
        # outvalues -> list of np arrays (shape = (layer_size, batchsize)). The ith entry contains the
        # output values of the neurons for example in the batch as its columns. 
        # outderivs -> similar to outvalues. But has derivatives of the output values wrt activation
        # value. here, activation value = WX + B
        outvalues, outderivs = [np.copy(input)], [None]

        # loop to compute outvalues and outderiv entries for all hidden layers.
        for idx in range(1, self.hlayercount + 1):
            next_act_values = ((weights[idx] @ input).T + biases[idx]).T # transposing is done to resolve broadcast issues
            next_out_values = self.hidden_actfn[idx][0](next_act_values) # apply act function to get outvalues entry
            next_out_derivs = self.hidden_actfn[idx][1](next_out_values, next_act_values) # apply act function derivative to get outderivs entry
            outvalues.append(next_out_values)
            outderivs.append(next_out_derivs)
            input = next_out_values
        
        # final layer - get act values and compute loss by using the loss object.
        finidx = self.hlayercount + 1
        fin_act_values = ((weights[finidx] @ input).T + biases[finidx]).T
        Loss = self.loss_fn
        y_pred, loss = Loss.compute_loss_and_final_layer_gradients_preact(fin_act_values, labels, False)
        # return necessary data for backward and computing acc, loss metrics.
        return outvalues, outderivs, fin_act_values, y_pred, loss

    # implements the backward pass for a batch to compute gradients
    # fin_act_values -> final_layer_act_values (shape = (out_dim, batchsize)); outvalues and outderivs are returned by
    # forward -> to save computation cost; labels -> list of labels of size batchsize
    def backward(self, weights, biases, labels, outvalues, outderivs, fin_act_values):
        # lists to store loss gradients wrt weights and biases for each layer
        weight_gradients, bias_gradients = [None for i in range(self.hlayercount+2)], [None for i in range(self.hlayercount+2)] # to be returned

        Loss = self.loss_fn
        # get loss gradients wrt final layer activation value using the loss function
        # gradients_act_values = gradient of loss wrt activation values of final layer here.
        _, _, gradients_act_values = Loss.compute_loss_and_final_layer_gradients_preact(fin_act_values, labels, True)
        # weight gradient(wg) for a layer is now a 3-d matrix. wg[a,b,c] = d Loss / d w_{ab} for the cth example in the batch
        # w_{ab} has a for the current layer and b for the previous layer. Clearly, we have 
        # wg[a,b,c] = gradients_act_values[a,c] * outvalues(of prev layer)[b,c] as indicated in the einsum subscripts
        # here, wg[:,:,c] gives the loss grad weights for the cth example.
        weight_gradients[self.hlayercount + 1]  = np.einsum('ac,bc->abc', gradients_act_values, outvalues[self.hlayercount])
        bias_gradients[self.hlayercount + 1] = np.copy(gradients_act_values) # bias gradients are just a copy of gradients_act_values

        # loop to backprop through the hidden layers
        for idx in range(self.hlayercount, 0, -1):
            # loss_grad_outputs = gradient of loss wrt output of current layer(index = idx) is computed
            loss_grad_outputs = np.transpose(weights[idx + 1]) @ (gradients_act_values)
            # gradients_act_values = gradient of loss wrt activation values of layer idx is computed
            gradients_act_values = np.multiply(loss_grad_outputs, outderivs[idx])
            # gradient of loss wrt weights and biases of layer idx are computed as usual
            weight_gradients[idx]  = np.einsum('ac,bc->abc', gradients_act_values, outvalues[idx-1])
            bias_gradients[idx] = np.copy(gradients_act_values)
        
        # lists containing loss gradients wrt weights and biases for every layer is returned
        # this will be aggregated (over batch examples) in train() and updates will be made 
        # with the optimizer
        return weight_gradients, bias_gradients

    # function to split X and y into batches of size batch_size. Required to exploit
    # power of numpy on matrix operations by "forwarding" and "backwarding" all examples
    # in a batch simultaneously. Final batch may not have its size as batchsize
    def make_batches(self, X, y, batch_size):
        batch_X, batch_y = [], []
        # loop to split the data into batches for exploiting numpy matrix ops speedup
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
        # return the data batches
        return batch_X, batch_y
    
    # function to test the current parameters of the network against validation/test data
    # again, these data must be in batches
    def test(self, weights, biases, batches, tname):
        total_count = 0
        total_correct = 0
        total_loss = 0.0
        # loop through all batches and keep count of correct predictions and total loss
        for X, y in zip(batches[0], batches[1]):
            _, _, _, y_pred, loss = self.optimizer.forward(weights, biases, X, y)
            total_loss += loss
            pred_labels = np.argmax(y_pred, axis=0)
            total_count += len(y)
            total_correct += len(np.where(pred_labels == y)[0])
        # compute average loss and accuracy, print and return these metrics
        tacc, tloss = total_correct/total_count, total_loss/total_count
        print(f'{tname} accuracy = {tacc}; {tname} loss = {tloss}')
        return tacc, tloss

    # to log the test loss, test acc and test data confusion matrix in a wandb run
    def plot_confusion_matrix(self, weights, biases, testbatches):
        y_pred = []
        y_true = []
        total_loss = 0
        labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 
                                'Sneaker', 'Bag', 'Ankle boot']
        # compute pred labels, loss and acc on test data
        for X, y in zip(testbatches[0], testbatches[1]):
            _, _, _, batch_pred, loss = self.optimizer.forward(weights, biases, X, y)
            batch_pred = np.argmax(batch_pred, axis=0)
            total_loss += loss
            y_pred += list(batch_pred)
            y_true += list(y)
        y_pred = np.array(y_pred)
        y_true = np.array(y_true)
        test_loss = total_loss / len(y_pred)
        test_acc = np.mean(np.where(y_pred == y_true, 1, 0))
        # code for plotly styled confusion matrix with colorbar
        cf_matrix = confusion_matrix(y_true, y_pred, labels=np.arange(self.out_layer_size))
        fig = px.imshow(cf_matrix, labels=dict(x="Predicted", y="True Class", color="Count"),
                        x=labels, y=labels, title='Confusion Matrix', text_auto=True,
                        color_continuous_scale=px.colors.sequential.OrRd)
        fig.update_xaxes(side="top")
        # log both the confusion matrix from wandb.plot and plotly plot along with loss, acc
        wandb.log({"confusion matrix 1" : wandb.plot.confusion_matrix(preds=y_pred, y_true=y_true, class_names=labels),
                    "confusion matrix 2" : fig,
                    "test loss" : test_loss,
                    "test accuracy" : test_acc})
        return test_acc, test_loss 

    # function to shuffle the train batches before each epoch to avoid overfitting.
    # increased val accuracy levels to > 89.2 % for good hyperparameter combinations.
    # important to do this. 
    def shuffle_train(self, train_batches):
        X, y = train_batches
        perm = self.shf_rg.permutation(len(X))
        return [X[p] for p in perm], [y[p] for p in perm]

    # function to train the neural network using train_data and perform validation
    # (for hyperparameter fine-tuning) using val_data
    def train(self, train_data, val_data, epochs, batchsize, learning_rate, silent=True, log_wandb=False):
        # batchify the train and validation data
        (train_X, train_y) = train_data
        (val_X, val_y) = val_data
        train_batches = self.make_batches(train_X, train_y, batchsize)
        val_batches = self.make_batches(val_X, val_y, batchsize)
        self.shf_rg = np.random.RandomState(42) # random state for determinism when shuffling train batches

        weights, biases = self.init_parameters() # initialize weight and bias parameters
        for e in range(epochs):
            # forward and backward over all data (1 epoch)
            counter = 0 # for non-silent mode; we will calculate val loss and val acc every 100 train batches; this will be printed
            train_batches = self.shuffle_train(train_batches) # shuffle train batches
            for X, y in tqdm(zip(train_batches[0], train_batches[1]), total=len(train_batches[0])):
                counter += 1
                # do forward and backward pass for 1 batch
                batch_weight_gradient, batch_bias_gradient = [None for i in range(self.hlayercount+2)], [None for i in range(self.hlayercount+2)]
                outvalues, outderivs, fin_act_values, y_pred, loss = self.optimizer.forward(weights, biases, X, y)
                weight_gradients, bias_gradients = self.optimizer.backward(weights, biases, y, outvalues, outderivs, fin_act_values)

                # aggregate the gradients for all the examples in the batch
                # and add the L2 regularization terms
                for idx in range(1, self.hlayercount+2):
                    batch_bias_gradient[idx] = np.sum(bias_gradients[idx], axis=-1)
                    batch_weight_gradient[idx] = np.sum(weight_gradients[idx], axis=-1)

                    # L2 regularization -> one update for every optimizer step
                    batch_bias_gradient[idx] += self.weight_decay * biases[idx]
                    batch_weight_gradient[idx] += self.weight_decay * weights[idx]

                # make log and test after every 100 updates if silent is False
                self.optimizer.update_parameters(weights, biases, learning_rate, batch_weight_gradient, batch_bias_gradient)
                if (not silent and counter == 100):
                    _, _ = self.test(weights, biases, val_batches, 'cur validation')
                    counter = 0
            
            # calculate the train, validation losses and accuracies for the epoch
            # and log them if log_wandb is True
            print(f'epoch - {e+1}')
            train_acc, train_loss = self.test(weights, biases, train_batches,'train')
            validation_acc, validation_loss = self.test(weights, biases, val_batches,'validation')
            if log_wandb:
                wandb.log({'epoch' : e+1, 
                           'train_loss' : train_loss,
                           'train_acc' : train_acc,
                           'validation_loss' : validation_loss,
                           'validation_acc' : validation_acc
                           })
        # return the trained network's weights and biases
        return weights, biases
