import numpy as np
from activation_functions import *
from neural_network import *
import wandb
from sklearn.model_selection import train_test_split
from keras.datasets import fashion_mnist, mnist
import argparse as ap

# function to create a parser that parses the commandline arguments
def gen_parser():
    parser = ap.ArgumentParser(description='Sweeping Code')
    parser.add_argument('-d', '--dataset', dest='dataset', default='fashion_mnist', choices=['fashion_mnist', 'mnist'], help='Dataset to be used')
    parser.add_argument('-sid', '--sweep_id', dest='sid', default=None, help='sweep id for sweep agent')
    return parser

# direct zero-one normalization by dividing by 255. 
def simple_normalize(x):
    return x/255

def agent_code():
    run = wandb.init(reinit=True)
    wargs = wandb.config
    if wargs.dataset == 'fashion_mnist':
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    else:
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = simple_normalize(x_train.astype(np.float64))
    train_X, val_X, train_y, val_y = train_test_split(x_train, y_train, test_size=0.1, shuffle=True, random_state=42)
    train_X, val_X = train_X.reshape(len(train_X), -1), val_X.reshape(len(val_X),-1)

    # change run name appropriately
    name1 = f'e={wargs.epochs}_hlc={wargs.num_hlayers}_hs={wargs.hidden_size}_bs={wargs.batch_size}'
    name2 = f'wd={wargs.weight_decay}_wi={wargs.weight_init}_loss={wargs.loss}_lr={wargs.learning_rate}'
    name3 = f'act={wargs.activation_function}_opt={wargs.optimizer}_ds={wargs.dataset}'
    run.name = f'{name1}_{name2}_{name3}'
    
    nn = NeuralNetwork(wargs, 10, train_X.shape[1])
    _, _ = nn.train((train_X, train_y), (val_X, val_y), wargs.epochs, wargs.batch_size, wargs.learning_rate, silent=True,log_wandb=True)

# change default values to reflect best results later
if __name__ == '__main__':
    wandb.login()
    args = gen_parser().parse_args()
    if (args.sid == None):
        primary_sweep_config = {'method' : 'random',
                        'name' : 'primary-sweep',
                        'metric' : {
                                'goal' : 'minimize',
                                'name' : 'validation_acc'
                            }, 
                            'parameters': {
                                'epochs' :  {'values' : [5, 10, 15], 'probabilities' : [0.3, 0.5, 0.2]},
                                'num_hlayers' : {'values' : [3, 4, 5], 'probabilities' : [0.25, 0.25, 0.5]},
                                'hidden_size' : {'values' : [32, 64, 128], 'probabilities' : [0.3, 0.3, 0.4]},
                                'weight_decay' : {'values' : [0.0, 0.05, 0.1, 0.5]},
                                'learning_rate' : {'values' : [1e-4, 4e-4, 8e-4, 16e-4]},
                                'batch_size' : {'values' : [16, 32, 64, 128]},
                                'loss' : {'value' : 'cross_entropy'},
                                'optimizer' : {'values' : ['sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam']},
                                'weight_init' : {'values' : ['random', 'xavier'], 'probabilities' : [0.3, 0.7]},
                                'activation_function' : {'values' : ['sigmoid', 'tanh', 'relu'], 'probabilities' : [0.3, 0.3, 0.4]},
                                'momentum' : {'value' : 0.9},
                                'beta' : {'value' : 0.95},
                                'beta1' : {'value' : 0.9},
                                'beta2' : {'value' : 0.999},
                                'epsilon' : {'value' : 1e-8},
                                'dataset' : {'value' : args.dataset},
                            }
                        }
        sec_sweep_config = {'method' : 'random',
                'name' : 'secondary-sweep',
                'metric' : {
                        'goal' : 'minimize',
                        'name' : 'validation_acc'
                    }, 
                    'parameters': {
                        'epochs' :  {'values' : [10,15,18]},
                        'num_hlayers' : {'value' : 5},
                        'hidden_size' : {'values' : [64, 128]},
                        'weight_decay' : {'values' : [0.0, 0.05, 0.1, 0.25, 0.5]},
                        'learning_rate' : {'values' : [1e-4, 4e-4, 8e-4, 16e-4, 32e-4]},
                        'batch_size' : {'values' : 128},
                        'loss' : {'value' : 'cross_entropy'},
                        'optimizer' : {'values' : ['sgd', 'nag', 'rmsprop', 'adam']},
                        'weight_init' : {'values' : ['he', 'xavier'], 'probabilities' : [0.35, 0.65]},
                        'activation_function' : {'values' : ['leakyrelu', 'elu', 'swish'], 'probabilities' : [0.4, 0.2, 0.4]},
                        'momentum' : {'value' : 0.9},
                        'beta' : {'value' : 0.95},
                        'beta1' : {'value' : 0.9},
                        'beta2' : {'value' : 0.999},
                        'epsilon' : {'value' : 1e-8},
                        'dataset' : {'value' : args.dataset},
                    },
                    'run_cap' : 300
                }
        sweep_id = wandb.sweep(entity='cs19b021', project='cs6910-assignment1',sweep=sec_sweep_config)
        print(sweep_id)
    else:
        wandb.agent(sweep_id=args.sid, entity='cs19b021', project='cs6910-assignment1', function=agent_code)
