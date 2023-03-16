# Code for wandb sweeping. Has 2 sweep configurations (1 primary, 1 secondary).
import numpy as np
from activation_functions import *
from neural_network import *
import wandb
from sklearn.model_selection import train_test_split
from keras.datasets import fashion_mnist, mnist
import argparse as ap

# function to create a parser that parses the commandline arguments and displays help
def gen_parser():
    parser = ap.ArgumentParser(description='Sweeping Code')
    parser.add_argument('-d', '--dataset', dest='dataset', default='fashion_mnist', choices=['fashion_mnist', 'mnist'], help='Dataset to be used')
    parser.add_argument('-sid', '--sweep_id', dest='sid', default=None, help='sweep id for sweep agent')
    return parser

# direct zero-one normalization by dividing by 255. 
def simple_normalize(x):
    return x/255

# code run by any wandb agent. For efficient parallelized search using multiple cpus.
def agent_code():
    run = wandb.init(reinit=True)
    wargs = wandb.config
    # load dataset
    if wargs.dataset == 'fashion_mnist':
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    else:
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # normalize train dataset and split train data into train data and validation data
    x_train = simple_normalize(x_train.astype(np.float64))
    train_X, val_X, train_y, val_y = train_test_split(x_train, y_train, test_size=0.1, shuffle=True, random_state=42)
    train_X, val_X = train_X.reshape(len(train_X), -1), val_X.reshape(len(val_X),-1)

    # change run name appropriately
    name1 = f'e={wargs.epochs}_hlc={wargs.num_hlayers}_hs={wargs.hidden_size}_bs={wargs.batch_size}'
    name2 = f'wd={wargs.weight_decay}_wi={wargs.weight_init}_loss={wargs.loss}_lr={wargs.learning_rate}'
    name3 = f'act={wargs.activation_function}_opt={wargs.optimizer}_ds={wargs.dataset}'
    run.name = f'{name1}_{name2}_{name3}'
    
    # create neural network and train it with logging in silent mode
    nn = NeuralNetwork(wargs, 10, train_X.shape[1])
    _, _ = nn.train((train_X, train_y), (val_X, val_y), wargs.epochs, wargs.batch_size, wargs.learning_rate, silent=True,log_wandb=True)

if __name__ == '__main__':
    wandb.login()
    args = gen_parser().parse_args()
    if (args.sid == None):
        # here, we just create a new sweep using the primary/secondary configurations and print the id
        primary_sweep_config = {'method' : 'random',
                        'name' : 'new-primary-sweep',
                        'metric' : {
                                'goal' : 'minimize',
                                'name' : 'validation_acc'
                            }, 
                            'parameters': {
                                'epochs' :  {'values' : [6, 12, 16]},
                                'num_hlayers' : {'values' : [3, 4, 5]},
                                'hidden_size' : {'values' : [32, 64, 128]},
                                'weight_decay' : {'values' : [0.0, 0.05]},
                                'learning_rate' : {'values' : [1e-4, 10e-4]},
                                'batch_size' : {'values' : [16, 32, 64, 128]},
                                'loss' : {'value' : 'cross_entropy'},
                                'optimizer' : {'values' : ['sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam']},
                                'weight_init' : {'values' : ['random', 'xavier', 'he']},
                                'activation_function' : {'values' : ['sigmoid', 'tanh', 'relu', 'leakyrelu', 'elu', 'swish']},
                                'momentum' : {'value' : 0.9},
                                'beta' : {'value' : 0.95},
                                'beta1' : {'value' : 0.9},
                                'beta2' : {'value' : 0.999},
                                'epsilon' : {'value' : 1e-8},
                                'dataset' : {'value' : args.dataset},
                            },
                            'run_cap' : 400
                        }
        sec_sweep_config = {'method' : 'grid',
                'name' : 'new-secondary-sweep',
                'metric' : {
                        'goal' : 'grid',
                        'name' : 'validation_acc'
                    }, 
                    'parameters': {
                        'epochs' :  {'value' : 20},
                        'num_hlayers' : {'value' : 5},
                        'hidden_size' : {'value' : 128},
                        'weight_decay' : {'values' : [0.02, 0.05, 0.08, 0.1, 0.15, 0.25]},
                        'learning_rate' : {'values' : [4e-4, 6e-4, 8e-4, 10e-4, 15e-4, 20e-4]},
                        'batch_size' : {'value' : 128},
                        'loss' : {'value' : 'cross_entropy'},
                        'optimizer' : {'values' : ['sgd', 'adam']},
                        'weight_init' : {'value' :'xavier'},
                        'activation_function' : {'values' : ['leakyrelu', 'tanh', 'swish']},
                        'momentum' : {'value' : 0.9},
                        'beta' : {'value' : 0.95},
                        'beta1' : {'value' : 0.9},
                        'beta2' : {'value' : 0.999},
                        'epsilon' : {'value' : 1e-8},
                        'dataset' : {'value' : args.dataset},
                    }
                }

        sweep_id = wandb.sweep(entity='cs19b021', project='cs6910-assignment1',sweep=sec_sweep_config)
        print(sweep_id)
    else:
        # here, a sweep_id was received in cmdline and therefore, a sweep agent is created to
        # train networks acc. to hyperparameter combinations received from wandb sweep server
        wandb.agent(sweep_id=args.sid, entity='cs19b021', project='cs6910-assignment1', function=agent_code)
