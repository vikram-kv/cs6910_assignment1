import argparse as ap


def gen_parser():
    parser = ap.ArgumentParser(description='Multi-Layer Feedfoward Neural Network with Various Optimizers')
    parser.add_argument('-wp','--wandb_project',dest='wdp', default='cs6910-assignment1', help='Project name used to track experiments in Weights & Biases dashboard')
    parser.add_argument('-we', '--wandb_entity',dest='wde', default='vikram-kv', help='Wandb Entity used to track experiments in the Weights & Biases dashboard')
    parser.add_argument('-d', '--dataset', dest='dataset', default='fashion_mnist', choices=['fashion_mnist', 'mnist'], help='Dataset to be used')
    parser.add_argument('-e', '--epochs', dest='epochs', default=20, type=int, help='Number of epochs to train neural network')
    parser.add_argument('-b', '--batch_size', dest='batch_size', default=8, type=int, help='Batch size used to train neural network')
    parser.add_argument('-l', '--loss', dest='loss', default='cross_entropy', choices=['mean_squared_error', 'cross_entropy'], help='Loss function to be used for training')
    parser.add_argument('-o', '--optimizer', dest='opt', default='sgd', choices=['sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam'], help='Optimization method to use for training')
    parser.add_argument('-lr','--learning_rate', dest='lr', default=2*1e-2, type=float, help='Learning rate used to optimize model parameters')
    parser.add_argument('-m', '--momentum',dest='mom', default=0.5, type=float, help='Momentum used by momentum and nag optimizers')
    parser.add_argument('-beta', '--beta', dest='beta', default=0.5, type=float, help='Beta used by rmsprop optimizer')
    parser.add_argument('-beta1', '--beta1', dest='beta1', default=0.5, type=float, help='Beta1 used by adam and nadam optimizers.')
    parser.add_argument('-beta2', '--beta2', dest='beta2', default=0.5, type=float, help='Beta2 used by adam and nadam optimizers.')
    parser.add_argument('-eps', '--epsilon', dest='eps', default=0.000001, type=float, help='Epsilon used by optimizers')
    parser.add_argument('-w_d', '--weight_decay', dest='w_d', default=.0, type=float, help='Weight decay used by optimizers')
    parser.add_argument('-w_i', '--weight_init', dest='w_i', default='random', choices=['random', 'Xavier'], help='Weight initialization method to be used')
    parser.add_argument('-nhl', '--num_layers', dest='num_hidden_layers', default=1, type=int, help='Number of hidden layers used in feedforward neural network')
    parser.add_argument('-sz', '--hidden_size', dest='hidden_size', default=4, type=float, help='Number of hidden neurons in a feedforward layer')
    parser.add_argument('-a', '--activation', dest='act_func', default='sigmoid', choices=['identity', 'sigmoid', 'tanh', 'ReLU'], help='Activation function to be used')
    return parser