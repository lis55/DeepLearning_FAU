from Layers import *
import numpy as np
from copy import *
import pickle
from Layers import Base
from Optimization import *
import os

class NeuralNetwork(object):

    def __init__(self, optimizer, weights_initializer, bias_initializer):
        self.layers = []  # 3 object
        self.loss = []
        self.data_layer = None  # net.data_layer = Helpers.IrisData()
        self.loss_layer = None  # net.loss_layer = SoftMax.SoftMax()

        self.optimizer = deepcopy(optimizer)
        self.weights_initializer = deepcopy(weights_initializer)
        self.bias_initializer = deepcopy(bias_initializer)
        self.phase = None
        self.regularization_loss = 0

    def append_trainable_layer(self, layer):
        layer.initialize(self.weights_initializer, self.bias_initializer)
        layer.optimizer = deepcopy(self.optimizer)
        self.layers.append(layer)

    def forward(self):
        self.loss = []
        self.input_tensor, self.label_tensor = self.data_layer.forward()
        input_tensor = copy(self.input_tensor)
        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)  #
            if isinstance(layer, FullyConnected.FullyConnected or Conv.Conv):
                if self.optimizer is not None:
                    base_class = Base.Base_class()
                    base_class.regularizer = Constraints.L2_Regularizer(4e-4)
                    self.regularization_loss += base_class.calculate_regularization_loss(layer)
                    print('regularization_loss', self.regularization_loss)

        self.loss_out = self.loss_layer.forward(input_tensor, self.label_tensor) + self.regularization_loss
        self.loss.append(self.loss_out)  # rerurn loss=sum of -log(yi)
        return self.loss_out

    def backward(self):
        error_tensor = self.loss_layer.backward(self.label_tensor)
        for layer in reversed(self.layers):
            error_tensor = layer.backward(error_tensor)

        pass

    def train(self, iterations):
        for iter in range(iterations):
            self.forward()
            self.backward()
        pass

    def test(self, input_tensor):
        self.set_phase(Base.Phase.test)
        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)
        prediction = self.loss_layer.predict(input_tensor)
        return prediction

    def set_phase(self, phase):
        for layer in self.layers:
            if isinstance(layer, Dropout.Dropout):
                layer.phase = phase
            if isinstance(layer, BatchNormalization.BatchNormalization):
                layer.phase = phase

    def del_data_layer(self):
        self.data_layer = None

    def set_data_layer(self, data_layer):
        self.data_layer = data_layer

def save(filename, net):
    dir, filename = os.path.split(filename)
    if not os.path.exists(dir):
        os.makedirs(dir)

    temp = net.data_layer
    with open(filename, "wb") as f:
        net.del_data_layer()
        pickle.dump(net, f)
        net.data_layer = temp

def load(filename, data_layer):
    net = pickle.load(open(filename, "rb"))
    net.data_layer = data_layer
    return net


