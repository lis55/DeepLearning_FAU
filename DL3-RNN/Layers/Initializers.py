import numpy as np
import scipy as sc


class Constant:

    def __init__(self, const):
        self.const = const

    def initialize(self, weight, shape1, shape0):
        if (len(weight) == 2):
            weights= np.zeros((shape1, shape0))
            initialized_tensor = np.ones_like(weights) * self.const
        else:
            initialized_tensor = np.ones_like(weight) * self.const
        return initialized_tensor


class UniformRandom:

    def initialize(self,  weight, shape1, shape0):
        if (len(weight)==2):
            weights = np.zeros((shape1, shape0))
            initialized_tensor = np.random.uniform(0, 1, weights.shape)
        else:
            initialized_tensor = np.random.uniform(0, 1, weight.shape)
        return initialized_tensor


class Xavier:

    def initialize(self, weight, shape1, shape0):
        fan_in = 0
        fan_out = 0
        print('weights',weight)
        if (len(weight) == 2):  # weight is 2d tuple
            weights = np.zeros((shape1, shape0))
            fan_out = weights.shape[1]
            fan_in = weights.shape[0]
        else:
            weights = np.zeros_like(weight)
            fan_in = weights.shape[1] * weights.shape[2] * weights.shape[3]
            fan_out = weights.shape[0] * weights.shape[2] * weights.shape[3]
        sigma = sc.sqrt(2/(fan_in + fan_out))
        mu = 0
        initialized_tensor = np.random.normal(mu, sigma, weights.shape)
        return initialized_tensor


class He():

    def __init__(self):
        pass

    def initialize(self, weight, shape1, shape0):
        fan_in = 0

        if (len(weight) == 2):
            weights = np.zeros((shape1, shape0))
            fan_in = weights.shape[0]
        else:
            weights = np.zeros_like(weight)
            fan_in=np.prod(weights.shape[1:])
        sigma = sc.sqrt(2/(fan_in ))
        mu = 0
        initialized_tensor = np.random.normal(mu, sigma, weights.shape)

        return initialized_tensor


