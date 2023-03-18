# Code for running 3 hyperparameter combinations for the mnist dataset
import numpy as np
from activation_functions import *
from neural_network import *
import wandb
from sklearn.model_selection import train_test_split
from keras.datasets import mnist

# direct zero-one normalization by dividing by 255. 
def simple_normalize(x):
    return x/255

if __name__ == '__main__':
    wandb.login()
    # define our 3 configurations for finetuning hyperparameters for mnist dataset
    config1 = {'epochs' :  12,
                'batch_size' : 128,
                'loss' : 'cross_entropy',
                'optimizer' : 'adam',
                'learning_rate' : 0.001,
                'momentum' : 0.9,
                'beta' : 0.95,
                'beta1' :  0.9,
                'beta2' :  0.999,
                'epsilon' :  1e-8,
                'weight_decay' : 0.02,
                'weight_init' : 'xavier',
                'num_hlayers' : 5,
                'hidden_size' : 128,
                'activation_function' : 'tanh',
                'dataset': 'mnist',
                }
    config2 = {'epochs' :  12,
                'batch_size' : 16,
                'loss' : 'cross_entropy',
                'optimizer' : 'sgd',
                'learning_rate' : 0.001,
                'momentum' : 0.9,
                'beta' : 0.95,
                'beta1' :  0.9,
                'beta2' :  0.999,
                'epsilon' :  1e-8,
                'weight_decay' : 0.0,
                'weight_init' : 'xavier',
                'num_hlayers' : 5,
                'hidden_size' : 128,
                'activation_function' : 'tanh',
                'dataset': 'mnist',
                }
    config3 = {'epochs' :  12,
                'batch_size' : 128,
                'loss' : 'cross_entropy',
                'optimizer' : 'adam',
                'learning_rate' : 0.0004,
                'momentum' : 0.9,
                'beta' : 0.95,
                'beta1' :  0.9,
                'beta2' :  0.999,
                'epsilon' :  1e-8,
                'weight_decay' : 0.05,
                'weight_init' : 'xavier',
                'num_hlayers' : 5,
                'hidden_size' : 128,
                'activation_function' : 'leakyrelu',
                'dataset': 'mnist',
                }
    configs = [config1, config2, config3]
    # load mnist dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # normalize and split train data into train data and validation data
    x_train = simple_normalize(x_train.astype(np.float64))
    train_X, val_X, train_y, val_y = train_test_split(x_train, y_train, test_size=0.1, shuffle=True, random_state=42)
    train_X, val_X = train_X.reshape(len(train_X), -1), val_X.reshape(len(val_X),-1)

    # normalize and reshape the test data
    x_test = simple_normalize(x_test.astype(np.float64))
    x_test = x_test.reshape(len(x_test), -1)

    for i in [0,1,2]:
        conf = configs[i]
        # init wandb run and update run name
        with wandb.init(entity='cs19b021', project='cs6910-assignment1', config=conf,tags=['mnist_run']) as run:
            wargs = wandb.config
            nn = NeuralNetwork(wargs, 10, train_X.shape[1])
            run.name = f'mnist-recommendation{i+1}'
            # we will not log training metrics. We log only the confusion matrix
            weights, biases = nn.train((train_X, train_y), (val_X, val_y), wargs.epochs, wargs.batch_size, wargs.learning_rate, silent=True, log_wandb=True)

            # run the model on test data and log test metrics
            testbatches = nn.make_batches(x_test, y_test, wargs.batch_size)
            tacc, tloss = nn.test(weights, biases, testbatches, 'test')
            wandb.log({"test loss" : tloss, "test accuracy" : tacc})