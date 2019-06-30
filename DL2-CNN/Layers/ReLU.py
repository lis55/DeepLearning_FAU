import numpy as np


class ReLU:
    def __init__(self):
        self.input_tensor = None

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        return np.maximum(input_tensor, 0)

    def backward(self, error_tensor):
        error_tensor_old = error_tensor.copy()
        error_tensor[self.input_tensor <= 0] = 0
        error_tensor[self.input_tensor > 0] = 1
        return np.multiply(error_tensor, error_tensor_old)