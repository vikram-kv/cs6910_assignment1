# CS6910 Assignment 1
First Assignment of the Deep Learning Course (CS6910), Summer 2023.

**Wandb Report** : [link](https://wandb.ai/cs19b021/cs6910-assignment1/reports/CS6910-Assignment-1--VmlldzozNzU3NDkz)

# Code Details

## Files

1. **activation_functions.py** - Contains the definitions of tanh, relu, sigmoid, linear, leaky relu, elu activation functions and their derivatives.
2. **explore_dataset.py** - Contains code for saving representative images of each class for both the datasets - mnist and fashion_mnist - in wandb project.
3. **loss_functions.py** - Contains classes for MSE Loss and Cross Entropy Loss that have members for loss computation and loss gradient (with respect to final layer activation value). Throughout the code, activation value = value of input to the activation function.
4. **main.py** - Contains code for parsing commandline arguments, loading dataset, performing train/validation split and running the model. Wandb logging is done optionally.
5. **neural_network.py** - Contains implementation of the neural network with forward(), backward(), train() and test() functions. The most important file in the code. Initialization techniques implemented are Random initialization, He initialization and Xavier initialization.
6. **optimizers.py** - Contains the various optimizers supported by the project. These are sgd, momentum, nag, rmsprop, adam and nadam.
7. **.ipynb files** - redundant files. Were used to test initial ideas easily.
