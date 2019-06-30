import numpy as np


class Constant:
    def __init__(self, con):#constant initializer
        self.con = con

    def initialize(self, weights_shape, fan_in, fan_out):#fan_in/out dimension of the weights
        weights = np.zeros(weights_shape) + self.con 
        return weights


class UniformRandom:
    def __init__(self):
        pass

    def initialize(self, weights_shape, fan_in, fan_out):
        # shape = np.shape(weights)
        num = np.prod(weights_shape)
        weights = np.random.rand(num)
        weights = weights.reshape(weights_shape)
        return weights


class Xavier:
    def __init__(self):
        pass

    def initialize(self, weights_shape, fan_in, fan_out):
        weights = np.random.normal(0.0, np.sqrt(2.0 / (fan_in + fan_out)),
                                   weights_shape)  # normal gaussian distribution
        return weights


class He:
    def __init__(self):
        pass

    def initialize(self, weights_shape, fan_in, fan_out):
        weights = np.random.normal(0.0, np.sqrt(2.0 / fan_in), weights_shape)
        return weights
