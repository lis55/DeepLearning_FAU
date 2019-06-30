import numpy as np
from copy import *
from enum import Enum
from Layers import FullyConnected
from Layers import Conv
from Layers import RNN
from Layers import LSTM


class Base_class():

    def __init__(self):
        self.regularizer = None
        self.phase = None

    def add_regularizer(self, regularizer):
        self.regularizer = regularizer

    def calculate_regularization_loss(self, layer):
        regularization_loss = 0
        print('self.regularizer',self.regularizer)

        if isinstance(layer,FullyConnected.FullyConnected):
            regularization_loss = self.regularizer.norm(layer.weights)
            print('000',regularization_loss)

        if isinstance(layer,Conv.Conv):
            regularization_loss = self.regularizer.norm(layer.weights)
            print('1111',regularization_loss)

        if isinstance(layer, RNN.RNN):
            regularization_loss = self.regularizer.norm(np.delete(layer.get_weights(),
                                                                    layer.get_weights().shape[1]-1, axis=1))
            print('2222',regularization_loss)

        if isinstance(layer, LSTM.LSTM):
            regularization_loss = self.regularizer.norm(np.delete(layer.get_weights(),
                                                                    layer.get_weights().shape[1]-1, axis=1))

        return regularization_loss


class Phase(Enum):
    train = 0
    test = 1
    validation = 2



