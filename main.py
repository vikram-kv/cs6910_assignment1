import numpy as np
import random
from helper_functions import *
from neural_network import *
import argparser
import wandb
from keras.datasets import fashion_mnist

def normalize_pixels(input : np.array):
    return input / 255

# change default values to reflect best results later
if __name__ == '__main__':
    parser = argparser.gen_parser()
    args = parser.parse_args()
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    train_X, train_y = [], []
    for x, y in zip(x_train, y_train):
        train_X += [normalize_pixels(x.flatten())]
        train_y += [y]
    
    random.seed(42)
    shuffle_list = list(zip(train_X, train_y))
    random.shuffle(shuffle_list)

    train_size = len(shuffle_list)
    lim = train_size // 10
    
    train_data = []
    val_data = []
    for i in range(len(shuffle_list)):
        if (i < lim):
            val_data += [shuffle_list[i]]
        else:
            train_data += [shuffle_list[i]]
    
    nn = NeuralNetwork(args, 10, x_train[0].size)
    nn.train(train_data, val_data, args.epochs, args.batch_size, args.lr)
    