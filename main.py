import numpy as np
from activation_functions import *
from neural_network import *
import wandb
from sklearn.model_selection import train_test_split
from keras.datasets import fashion_mnist, mnist

import argparse as ap

# function to create a parser that parses the commandline arguments
def gen_parser():
    parser = ap.ArgumentParser(description='Multi-Layer Feedfoward Neural Network with Various Optimizers')
    parser.add_argument('-wp','--wandb_project',dest='wdp', default='cs6910-assignment1', help='Project name used to track experiments in Weights & Biases dashboard')
    parser.add_argument('-we', '--wandb_entity',dest='wde', default='vikram-kv', help='Wandb Entity used to track experiments in the Weights & Biases dashboard')
    parser.add_argument('-d', '--dataset', dest='dataset', default='fashion_mnist', choices=['fashion_mnist', 'mnist'], help='Dataset to be used')
    parser.add_argument('-e', '--epochs', dest='epochs', default=10, type=int, help='Number of epochs to train neural network')
    parser.add_argument('-b', '--batch_size', dest='batch_size', default=64, type=int, help='Batch size used to train neural network')
    parser.add_argument('-l', '--loss', dest='loss', default='cross_entropy', choices=['mean_squared_error', 'cross_entropy'], help='Loss function to be used for training')
    parser.add_argument('-o', '--optimizer', dest='optimizer', default='adam', choices=['sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam'], help='Optimization method to use for training')
    parser.add_argument('-lr','--learning_rate', dest='learning_rate', default=1e-4, type=float, help='Learning rate used to optimize model parameters')
    parser.add_argument('-m', '--momentum', dest='momentum', default=0.9, type=float, help='Momentum used by momentum and nag optimizers')
    parser.add_argument('-beta', '--beta', dest='beta', default=0.95, type=float, help='Beta used by rmsprop optimizer')
    parser.add_argument('-beta1', '--beta1', dest='beta1', default=0.9, type=float, help='Beta1 used by adam and nadam optimizers.')
    parser.add_argument('-beta2', '--beta2', dest='beta2', default=0.999, type=float, help='Beta2 used by adam and nadam optimizers.')
    parser.add_argument('-eps', '--epsilon', dest='epsilon', default=1e-8, type=float, help='Epsilon used by optimizers')
    parser.add_argument('-w_d', '--weight_decay', dest='weight_decay', default=0.0, type=float, help='Weight decay used by optimizers')
    parser.add_argument('-w_i', '--weight_init', dest='weight_init', default='xavier', choices=['random', 'xavier'], help='Weight initialization method to be used')
    parser.add_argument('-nhl', '--num_layers', dest='num_layers', default=5, type=int, help='Number of hidden layers used in feedforward neural network')
    parser.add_argument('-sz', '--hidden_size', dest='hidden_size', default=128, type=int, help='Number of hidden neurons in a feedforward layer')
    parser.add_argument('-a', '--activation', dest='activation', default='tanh', choices=['identity', 'sigmoid', 'tanh', 'relu', 'leakyrelu', 'elu'], help='Activation function to be used')
    parser.add_argument('-l', '--log_data', dest='logging', type=int, default=0, help='non-zero for wandb sweeping and logging')
    return parser

def normalize_pixels(input : np.array):
    return input / 255

# change default values to reflect best results later
if __name__ == '__main__':
    parser = gen_parser()
    args = parser.parse_args()
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    x_train = x_train.astype(np.float64)
    x_test = x_test.astype(np.float64)
    x_train, x_test = x_train.reshape(x_train.shape[0], -1), x_test.reshape(x_test.shape[0], -1)
    x_train /= 255
    x_test /= 255
    train_X, val_X, train_y, val_y = train_test_split(x_train, y_train, test_size=0.1, shuffle=True, random_state=1)
    
    nn = NeuralNetwork(args, 10, train_X.shape[1])
    nn.train((train_X, train_y), (val_X, val_y), args.epochs, args.batch_size, args.learning_rate)
    