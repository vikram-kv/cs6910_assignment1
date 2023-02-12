import numpy as np

class ActivationFunction():
    def value(self, *args):
        pass
    def derivative(self, *args):
        pass
    @property
    def name(self):
        pass

class Sigmoid(ActivationFunction):
    @property
    def name(self):
        return 'sigmoid'
    @staticmethod
    def value(x):
        return 1.0/(1.0 + np.exp(-x))
    @staticmethod
    def derivative(x):
        fx = Sigmoid.value(x)
        return (1.0 - fx) * fx

class ReLu(ActivationFunction):
    @property
    def name(self):
        return 'relu'
    @staticmethod
    def value(x):
        return (0.0 if x < 0 else x)
    @staticmethod
    def derivative(x):
        return (0.0 if x < 0 else 1.0)

class TanH(ActivationFunction):
    @property
    def name(self):
        return 'tanh'
    @staticmethod
    def value(x):
        y, z = np.exp(x), np.exp(-x)
        return (y - z)/(y + z)
    @staticmethod
    def derivative(x):
        fx = TanH.value(x)
        return (1.0 - fx) * (1.0 + fx)

class Linear(ActivationFunction):
    @property
    def name(self):
        return 'linear'
    @staticmethod
    def value(x):
        return x
    @staticmethod
    def derivative(x):
        return 1.0