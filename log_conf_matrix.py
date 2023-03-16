# code to generate and plot the confusion matrix for the test data of the fashion_mnist dataset
# the best hyperparameter combination has been hardcoded here.
import numpy as np
from activation_functions import *
from neural_network import *
import wandb
from sklearn.model_selection import train_test_split
from keras.datasets import fashion_mnist, mnist

# direct zero-one normalization by dividing by 255. 
def simple_normalize(x):
    return x/255

if __name__ == '__main__':
    wandb.login()
    # here, we define the config for the best hyperparameter set on fashion_mnist
    wconfig = {'epochs' :  20,
                'num_hlayers' : 5,
                'hidden_size' : 128,
                'weight_decay' : 0.05,
                'learning_rate' : 4e-4,
                'batch_size' : 128,
                'loss' : 'cross_entropy',
                'optimizer' : 'adam',
                'weight_init' : 'xavier',
                'activation_function' : 'leakyrelu',
                'momentum' : 0.9,
                'beta' : 0.95,
                'beta1' :  0.9,
                'beta2' :  0.999,
                'epsilon' :  1e-8,
                'dataset': 'fashion_mnist',
                }

    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    # normalize and split train data into train data and validation data
    x_train = simple_normalize(x_train.astype(np.float64))
    train_X, val_X, train_y, val_y = train_test_split(x_train, y_train, test_size=0.1, shuffle=True, random_state=42)
    train_X, val_X = train_X.reshape(len(train_X), -1), val_X.reshape(len(val_X),-1)

    # init wandb run and update run name
    with wandb.init(entity='cs19b021', project='cs6910-assignment1',config=wconfig,tags=['confusion_matrix']) as run:
        wargs = wandb.config
        nn = NeuralNetwork(wargs, 10, train_X.shape[1])
        run.name = f'best_confusion_matrix_run'
        # we will not log training metrics. We log only the confusion matrix
        weights, biases = nn.train((train_X, train_y), (val_X, val_y), wargs.epochs, wargs.batch_size, wargs.learning_rate, silent=True, log_wandb=False)

        # run the model on (normalized, reshaped) test data and log the confusion matrix
        print('Test Performance')
        x_test = simple_normalize(x_test.astype(np.float64))
        x_test = x_test.reshape(len(x_test), -1)
        testbatches = nn.make_batches(x_test, y_test, wargs.batch_size)
        tacc, tloss = nn.plot_confusion_matrix(weights, biases, testbatches)
        print(f'test loss = {tloss}; test accuracy = {tacc}')