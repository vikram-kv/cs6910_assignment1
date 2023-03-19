# the main file of the folder. Contains code to load data, normalize it (and possibly augment train data [code needs to be slightly changed for this]),
# create and train a neural network and finally, display its performance on test data.
# wandb logging of losses and accuracies during training is optional.
import numpy as np
from activation_functions import *
from neural_network import *
import wandb
from sklearn.model_selection import train_test_split
from keras.datasets import fashion_mnist, mnist

# the below modules are not used in the execution. But certain experiment functions reference them in this
# file
from sklearn import preprocessing
import argparse as ap
from tqdm import tqdm
from skimage.transform import AffineTransform
from skimage import transform
from scipy import ndimage

# function to create a parser that parses the commandline arguments. Default values are the values from
# the best hyperparameter combination found from wandb sweeps.
def gen_parser():
    parser = ap.ArgumentParser(description='Multi-Layer Feedfoward Neural Network with Various Optimizers')
    parser.add_argument('-wp','--wandb_project',dest='wdp', default='cs6910-assignment1', help='Project name used to track experiments in Weights & Biases dashboard')
    parser.add_argument('-we', '--wandb_entity',dest='wde', default='cs19b021', help='Wandb Entity used to track experiments in the Weights & Biases dashboard')
    parser.add_argument('-d', '--dataset', dest='dataset', default='fashion_mnist', choices=['fashion_mnist', 'mnist'], help='Dataset to be used')
    parser.add_argument('-e', '--epochs', dest='epochs', default=20, type=int, help='Number of epochs to train neural network')
    parser.add_argument('-b', '--batch_size', dest='batch_size', default=128, type=int, help='Batch size used to train neural network')
    parser.add_argument('-l', '--loss', dest='loss', default='cross_entropy', choices=['mean_squared_error', 'cross_entropy'], help='Loss function to be used for training')
    parser.add_argument('-o', '--optimizer', dest='optimizer', default='adam', choices=['sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam'], help='Optimization method to use for training')
    parser.add_argument('-lr','--learning_rate', dest='learning_rate', default=0.001, type=float, help='Learning rate used to optimize model parameters')
    parser.add_argument('-m', '--momentum', dest='momentum', default=0.9, type=float, help='Momentum used by momentum and nag optimizers')
    parser.add_argument('-beta', '--beta', dest='beta', default=0.95, type=float, help='Beta used by rmsprop optimizer')
    parser.add_argument('-beta1', '--beta1', dest='beta1', default=0.9, type=float, help='Beta1 used by adam and nadam optimizers.')
    parser.add_argument('-beta2', '--beta2', dest='beta2', default=0.999, type=float, help='Beta2 used by adam and nadam optimizers.')
    parser.add_argument('-eps', '--epsilon', dest='epsilon', default=1e-8, type=float, help='Epsilon used by optimizers')
    parser.add_argument('-w_d', '--weight_decay', dest='weight_decay', default=0.02, type=float, help='Weight decay used by optimizers')
    parser.add_argument('-w_i', '--weight_init', dest='weight_init', default='xavier', choices=['random', 'xavier', 'he'], help='Weight initialization method to be used')
    parser.add_argument('-nhl', '--num_layers', dest='num_hlayers', default=5, type=int, help='Number of hidden layers used in feedforward neural network')
    parser.add_argument('-sz', '--hidden_size', dest='hidden_size', default=128, type=int, help='Number of hidden neurons in a feedforward layer')
    parser.add_argument('-a', '--activation', dest='activation_function', default='tanh', choices=['identity', 'sigmoid', 'tanh', 'relu', 'leakyrelu', 'elu', 'swish'], help='Activation function to be used')
    parser.add_argument('-uw', '--log_wandb', dest='log_wandb', type=int, default=0, help='non-zero for wandb logging')
    parser.add_argument('-rn', '--run_name', dest='run_name', default=None, help='non-zero for wandb sweeping and logging')
    return parser

# direct zero-one normalization by dividing by 255. 
def simple_normalize(x):
    return x/255

# mean 0; var 1 normalization. Standard scale is not better than simple_normalize; so not used.
def standard_scale(X):
    X_new = X.copy().reshape(X.shape[0], -1)
    scaler = preprocessing.StandardScaler().fit(X_new)
    return scaler.transform(X_new).reshape(X.shape)

# Code for augmenting train data by affine transformations.
# Not that great in improving accuracy -> achieved ~ 87.5 % on fashion_mnist in 5 epochs
# Without convolution layers, representative features (like vert, hori edges) are not
# extracted. This could be the reason why this did not boost accuracy.
def augment_train_data(x_train, y_train, rd_seed, num_dup):
    x_train_augment = list(x_train.copy())
    y_train_augment = list(y_train.copy())
    rd = np.random.RandomState(rd_seed)
    for i in tqdm(range(len(x_train))):
        img = x_train[i]
        lbl = y_train[i]
        rot_angles = rd.uniform(low=-30,high=+30,size=num_dup)
        shear_factors = rd.uniform(low=-np.pi/5,high=+np.pi/5,size=num_dup)
        scale_factors = rd.uniform(low=0.75,high=1.25,size=num_dup)
        imgs = []
        cur=0
        for (af,shf,scf) in zip(rot_angles,shear_factors,scale_factors):
            at = AffineTransform(rotation=af, shear=shf, scale=scf)
            nimg = transform.warp(img, at, order=1, preserve_range=True,mode='wrap')
            nimg = ndimage.uniform_filter(nimg, size=5)
            cur+=1
            if (cur%2 == 1):
                nimg = nimg[:,::-1] # vertical flip
            imgs += [nimg]
        x_train_augment += imgs
        y_train_augment += [lbl for _ in range(num_dup)]
    
    x_train_augment = np.array(x_train_augment)
    y_train_augment = np.array(y_train_augment)
    perm = rd.permutation(len(x_train_augment))
    return x_train_augment[perm], y_train_augment[perm]

if __name__ == '__main__':
    parser = gen_parser()
    args = parser.parse_args()
    # load dataset based on choice
    if args.dataset == 'fashion_mnist':
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    else:
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # normalize and split train data into train data and validation data
    x_train = simple_normalize(x_train.astype(np.float64))
    train_X, val_X, train_y, val_y = train_test_split(x_train, y_train, test_size=0.1, shuffle=True, random_state=42)
    train_X, val_X = train_X.reshape(len(train_X), -1), val_X.reshape(len(val_X),-1)

    # create the neural network
    nn = NeuralNetwork(args, 10, train_X.shape[1])
    if args.log_wandb != 0:
        # here, the user wants to log the run into a wandb project
        wandb.login()
        # wandb run configuration to be displayed on the webpage
        wandb_config = {'epochs' :  args.epochs,
                        'num_hlayers' : args.num_hlayers,
                        'hidden_size' : args.hidden_size,
                        'weight_decay' : args.weight_decay,
                        'learning_rate' : args.learning_rate,
                        'batch_size' : args.batch_size,
                        'loss' : args.loss,
                        'optimizer' : args.optimizer,
                        'weight_init' : args.weight_init,
                        'activation_function' : args.activation_function,
                        'momentum' : args.momentum,
                        'beta' : args.beta,
                        'beta1' : args.beta1,
                        'beta2' : args.beta2,
                        'epsilon' : args.epsilon,
                        'dataset' : args.dataset,
                    }
        # init wandb run and update run name(if given)
        with wandb.init(project=args.wdp, entity=args.wde, config=wandb_config) as run:
            if args.run_name != None:
                run.name = args.run_name
                # train network
            weights, biases = nn.train((train_X, train_y), (val_X, val_y), args.epochs, args.batch_size, args.learning_rate, silent=False, log_wandb=True)
    else:
        # train network without wandb logging
        weights, biases = nn.train((train_X, train_y), (val_X, val_y), args.epochs, args.batch_size, args.learning_rate, silent=False, log_wandb=False)

    # run the model on (normalized, reshaped) test data and print performance details
    print('Test Performance')
    x_test = simple_normalize(x_test.astype(np.float64))
    x_test = x_test.reshape(len(x_test), -1)
    testbatches = nn.make_batches(x_test, y_test, args.batch_size)
    nn.test(weights, biases, testbatches, 'test')
    