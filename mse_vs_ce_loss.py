# Code for wandb sweeping. Has 2 sweep configurations (1 primary, 1 secondary).
import numpy as np
from activation_functions import *
from neural_network import *
import wandb
from sklearn.model_selection import train_test_split
from keras.datasets import fashion_mnist, mnist
import argparse as ap

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
    # here, we just create a new sweep using the primary/secondary configurations and print the id
    config1 = {'method' : 'grid',
                'name' : 'mse-ce-sweep1',
                'metric' : {
                        'goal' : 'minimize',
                        'name' : 'validation_acc'
                    }, 
                    'parameters': {
                        'epochs' :  {'value' : 5},
                        'num_hlayers' : {'value' : 4},
                        'hidden_size' : {'value' : 64},
                        'weight_decay' : {'value' : 0.05},
                        'learning_rate' : {'value' : 1e-3},
                        'batch_size' : {'value' : 32},
                        'loss' : {'values' : ['cross_entropy', 'mean_squared_error']},
                        'optimizer' : {'value' : 'adam'},
                        'weight_init' : {'value' : 'xavier'},
                        'activation_function' : {'value' : 'relu'},
                        'momentum' : {'value' : 0.9},
                        'beta' : {'value' : 0.95},
                        'beta1' : {'value' : 0.9},
                        'beta2' : {'value' : 0.999},
                        'epsilon' : {'value' : 1e-8},
                        'dataset' : {'value' : 'fashion_mnist'},
                    },
                }
    config2 = {'method' : 'grid',
            'name' : 'mse-ce-sweep2',
            'metric' : {
                    'goal' : 'minimize',
                    'name' : 'validation_acc'
                }, 
            'parameters': {
                    'epochs' :  {'value' : 5},
                    'num_hlayers' : {'value' : 4},
                    'hidden_size' : {'value' : 64},
                    'weight_decay' : {'value' : 0.05},
                    'learning_rate' : {'value' : 1e-3},
                    'batch_size' : {'value' : 32},
                    'loss' : {'values' : ['cross_entropy', 'mean_squared_error']},
                    'optimizer' : {'value' : 'sgd'},
                    'weight_init' : {'value' : 'random'},
                    'activation_function' : {'value' : 'sigmoid'},
                    'momentum' : {'value' : 0.9},
                    'beta' : {'value' : 0.95},
                    'beta1' : {'value' : 0.9},
                    'beta2' : {'value' : 0.999},
                    'epsilon' : {'value' : 1e-8},
                    'dataset' : {'value' : 'fashion_mnist'},
                },
            }
    configs = [config1, config2]
    for conf in configs:
        sweep_id = wandb.sweep(entity='cs19b021', project='cs6910-assignment1', sweep=conf)
        wandb.agent(sweep_id=sweep_id, entity='cs19b021', project='cs6910-assignment1', function=agent_code)





