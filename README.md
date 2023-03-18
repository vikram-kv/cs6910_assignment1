# CS6910 Assignment 1
First Assignment of the Deep Learning Course (CS6910), Summer 2023.

**Wandb Report** : [link](https://wandb.ai/cs19b021/cs6910-assignment1/reports/CS6910-Assignment-1--VmlldzozNzU3NDkz)
**Experimental Learnable LeakyRelu Framework** : [link](https://github.com/vikram-kv/cs6910_a1_experiment_learnableleakyrelu) - Achieves > 88% val acc in the run with the default values very quickly. Experimental as it has not been well-tested.

# Code Details

## Files

1. **activation_functions.py** - Contains the definitions of tanh, relu, sigmoid, linear, leaky relu, elu activation functions and their derivatives.
2. **explore_dataset.py** - Contains code for saving representative images of each class for both the datasets - mnist and fashion_mnist - in wandb project with tag = 'images'
3. **log_conf_matrix.py** - Code to generate the plotly confusion matrix, wandb default confusion matrix, test accuracy, test loss for the best hyperparameter combination and log it in wandb with tag = 'confusion_matrix'
4. **loss_functions.py** - Contains classes for MSE Loss and Cross Entropy Loss that have members for loss computation and loss gradient (with respect to final layer activation value). Throughout the code, activation value = value of input to the activation function.
5. **mnist_run.py** - trains the neural network and logs performance metrics in wandb for mnist dataset with 3 different hyperparameter combinations. All runs will have tag = 'mnist_run'.
6. **mse_vs_ce_loss.py** - logs the performance metrics for 2 hyperparameter combinations in 2 sweeps by comparing MSE loss and CE loss. One hyperparameter combination is the best combination from the sweep and the other is custom-designed.
7. **neural_network.py** - Contains implementation of the neural network with forward(), backward(), train() and test() functions. The most important file in the code. Initialization techniques implemented are Random initialization, He initialization and Xavier initialization. Also has functions for shuffling the train batches across epochs, and for generating and logging confusion matrix plots.
8. **optimizers.py** - Contains the various optimizers supported by the project. These are sgd, momentum, nag, rmsprop, adam and nadam.
9. **sweep_code.py** - Contains the code for starting a wandb sweep server. By default, a sweep server is started and a sweep_id is printed. For parallelism, when -sid `sweep_id` is supplied in cmdline, a sweep agent is started that receives, trains and tests a hyperparameter combination. Very useful for parallelizing with multiple Colab instances/CPUs.
10. **train.py** - Contains code for parsing commandline arguments, loading dataset, performing train/validation split and running the model. Wandb logging is done optionally. [See Usage] Additionally, contains a function for performing affine transforms-based dataset augmentation.

## Usage

The main code file is **train.py**. To get detailed help instruction with all the features, run 

```bash
python3 train.py -h
```

To run with the default hyperparameter settings on fashion_mnist dataset, run

```bash
python3 train.py
```

To run with the default hyperparameter setting on fashion_mnist dataset and log the performance metrics(loss, accuracy for train, val) and print the test metrics, run

```bash
python train.py --wandb_entity [yourwandbentityname] --wandb_project [yourprojectname] -uw 1 -rn [run_name]
```
Ensure that you are already logged into wandb locally with the credentials of [yourwandbentityname] and [run_name] is the name of the run in wandb dashboard.

Other hyperparameter combinations may be tested using the details from the help menu.
