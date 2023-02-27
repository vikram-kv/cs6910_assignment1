import numpy as np
import random
from activation_functions import *
from neural_network import *
import argparser
import wandb
from sklearn.model_selection import train_test_split
from keras.datasets import fashion_mnist

def normalize_pixels(input : np.array):
    return input / 255

# change default values to reflect best results later
if __name__ == '__main__':
    parser = argparser.gen_parser()
    args = parser.parse_args()
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    x_train = x_train.astype(np.float64)
    x_test = x_test.astype(np.float64)
    x_train, x_test = x_train.reshape(x_train.shape[0], -1), x_test.reshape(x_test.shape[0], -1)
    x_train /= 255
    x_test /= 255
    train_X, val_X, train_y, val_y = train_test_split(x_train, y_train, test_size=0.1, shuffle=True, random_state=1)
    
    nn = NeuralNetwork(args, 10, train_X.shape[1])
    nn.train((train_X, val_X, train_y, val_y), args.epochs, args.batch_size, args.learning_rate)
    