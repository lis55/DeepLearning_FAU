import numpy as np
from copy import *


class FullyConnected:

    input_tensor = np.empty((0, 0))
    output_tensor = np.empty((0, 0))
    error_tensor = np.empty((0, 0))
    gradient = np.empty((0, 0))
    error = np.empty((0, 0))
    optimizer = None

    def __init__(self,input_size,output_size,*delta):
        self.input_size=input_size
        self.output_size=output_size
        self.weights = np.random.uniform(low=0, high=1, size=(self.input_size+1, self.output_size))
        if len(delta)==1:
            self.delta=delta
        else:
            self.delta = 5e-1

    def get_weights(self):

        return self.weights

    def set_weights(self, weights):
        self.weights = weights

    def initialize(self, weights_initializer, bias_initializer):
        weights = weights_initializer.initialize(self.weights[:-1, :],None,None)
        bias = np.expand_dims(self.weights[-1, :], axis=0)
        bias = bias_initializer.initialize(bias,None,None)
        self.weights = np.concatenate((weights, bias), axis=0)

    def forward(self, input_tensor):
        add_one = np.ones((input_tensor.shape[0], 1))   # (batch_size,1)
        self.input_tensor=np.hstack((input_tensor, add_one))
        self.output_tensor = np.dot(self.input_tensor, self.weights)  # according to test is input*weight

        return self.output_tensor

    def backward(self, error_tensor):
        self.error_input=error_tensor
        self.error_tensor=np.dot(self.error_input,self.weights.T)
        self.error_tensor_out = np.delete(self.error_tensor, -1, axis=1)
        self.get_gradient_weights()
        if (self.optimizer):
            self.weights = self.optimizer.calculate_update(self.delta, self.weights, self.gradient)

        return self.error_tensor_out

    def get_gradient_weights(self):
        self.gradient=np.zeros_like(self.weights)
        self.gradient = np.dot(self.input_tensor.T,self.error_input)
        return self.gradient

    def set_optimizer(self, optimizer):
        self.optimizer = deepcopy(optimizer)

