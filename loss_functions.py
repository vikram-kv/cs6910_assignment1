import numpy as np

# All Loss functions must implement the function compute_loss_and_final_layer_gradients_preact that 
# takes final layer pre-activation values and then, computes the loss and its gradient
# Remember we are doing everything batchwise to speed things up

class CrossEntropyLoss:
    # applies the softmax function to input after shifting it first to avoid numerical errors
    # input is of shape (num_classes, batch_size)
    @staticmethod
    def safe_softmax(input):
        input -= np.max(input, axis=0)
        input = np.exp(input)
        pred_probabilities = input/(np.sum(input, axis=0))
        return pred_probabilities
    
    # converts the true class to 1-hot encoding.
    # labels of shape (batch_size, 1)
    def convert_label_to_vector(self, labels, y_pred):
        y_true = np.zeros_like(y_pred)
        for idx in range(len(labels)):
            y_true[labels[idx],idx] = 1
        return y_true

    # computes the loss for the final layer and the gradients of loss wrt pre-activation values. 
    # y_true = label of true class -> converted to 1-hot vector form here.
    # final_layer_act_values -> shape (activation_values_count, batch_size)
    def compute_loss_and_final_layer_gradients_preact(self, final_layer_act_values, labels, needgradients=False):
        y_pred = self.safe_softmax(final_layer_act_values)
        y_true = self.convert_label_to_vector(labels, y_pred)
        loss = -np.mean(np.log2(np.sum(np.multiply(y_pred, y_true), axis=0)))
        if not needgradients:
            return y_pred, loss
        gradient_pre_activation = y_pred - y_true
        return y_pred, loss, gradient_pre_activation
    
class MeanSquareErrorLoss:
    # applies the softmax function to input after shifting it first to avoid numerical errors
    # input is of shape (num_classes, batch_size)
    @staticmethod
    def safe_softmax(input):
        input -= np.max(input, axis=0)
        input = np.exp(input)
        pred_probabilities = input/(np.sum(input, axis=0))
        return pred_probabilities
    
    # converts the true class to 1-hot encoding.
    # labels of shape (batch_size, 1)
    def convert_label_to_vector(self, labels, y_pred):
        y_true = np.zeros_like(y_pred)
        for idx in range(len(labels)):
            y_true[labels[idx], idx] = 1
        return y_true

    # computes the loss for the final layer and the gradients of loss wrt pre-activation values. 
    # y_true = label of true class -> converted to 1-hot vector form here.
    # remember we have a 0.5 factor for ease of gradient computation in loss definition = 0.5 * average of squared diff
    # over all samples
    def compute_loss_and_final_layer_gradients_preact(self, final_layer_act_values, labels, need_gradients=False):
        y_pred = self.safe_softmax(final_layer_act_values)
        y_true = self.convert_label_to_vector(labels, y_pred)
        loss = np.mean(np.sum(np.square(y_pred - y_true), axis=0)) * 0.5
        if not need_gradients:
            return y_pred, loss
        
        # it can be shown that the gradients matrix is add_term - sub_term as defined below.
        # taking one example (1 column) to see this helps
        add_term = np.multiply(y_pred, y_pred - y_true)
        sub_term = np.multiply(y_pred, np.sum(add_term, axis=0))
        gradient_pre_activation = add_term - sub_term

        return y_pred, loss, gradient_pre_activation

# add custom loss function's entry here
def get_loss_by_name(name):
    if (name == 'cross_entropy'):
        return CrossEntropyLoss()
    elif (name== 'mean_squared_error'):
        return MeanSquareErrorLoss()
    else:
        raise('Loss Function Not implemented'); exit(-1)