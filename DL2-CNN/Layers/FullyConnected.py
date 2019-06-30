import numpy as np
from copy import *


class FullyConnected:
    delta = 1

    def __init__(self, input_size, output_size, learning_rate=delta):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.random_sample((input_size + 1, output_size))
        self.input_tensor = np.empty((0, 0))
        self.output_tensor = np.empty((0, 0))
        self.error_tensor = np.empty((0, 0))
        self.gradient = np.empty((0, 0))
        self.error = np.empty((0, 0))
        self.optimizer = None
        self.learning_rate = learning_rate

    def initialize(self, weights_initializer, bias_initializer):
        weights = weights_initializer.initialize(np.shape(self.weights[:-1, :]), np.shape(self.weights[:-1, :])[0],
                                                 np.shape(self.weights[:-1, :])[1])
        bias = np.expand_dims(self.weights[-1, :], axis=0)
        bias = bias_initializer.initialize(bias.shape, bias.shape[0], bias.shape[1])
        self.weights = np.concatenate((weights, bias), axis=0)

    def forward(self, input_tensor):
        a = np.ones((input_tensor.shape[0], 1))
        self.input_tensor = np.column_stack((input_tensor, a))
        self.output_tensor = np.dot(self.input_tensor, self.weights)
        return self.output_tensor

    def backward(self, error_tensor):
        self.error_tensor = copy(error_tensor)
        self.error = copy(error_tensor)
        self.error_tensor = np.dot(error_tensor, self.weights.T)
        self.error_tensor = np.delete(self.error_tensor, -1, 1)
        self.get_gradient_weights()
        if (self.optimizer):
            self.weights = self.optimizer.calculate_update( self.weights, self.gradient)
        return self.error_tensor

    def get_gradient_weights(self):
        self.gradient = np.dot(self.input_tensor.T, self.error)
        return self.gradient

    def set_optimizer(self, optimizer):
        self.optimizer = deepcopy(optimizer)
