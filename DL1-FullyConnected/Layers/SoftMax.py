import numpy as np
from copy import *

class SoftMax:

    def __init__(self):
        self.yk_hat = np.empty((0, 0)) 
        self.error_tensor = np.empty((0, 0)) 
        self.loss = 0

    def predict(self, input_tensor):
        yk_hat = copy(input_tensor)
        row_max = np.expand_dims(np.max(input_tensor, 1), 1) #np.max removes a dimension after taking the maximum from each row 
        yk_hat = yk_hat - row_max         #substracts the maximum
        yk_hat = np.exp(yk_hat) / np.expand_dims(np.sum(np.exp(yk_hat), 1), 1)
        return yk_hat

    def forward(self, input_tensor, label_tensor):
        loss = 0
        self.yk_hat = self.predict(input_tensor)
        loss_tensor = - np.log(self.yk_hat)
        loss = np.sum(loss_tensor[label_tensor==1])
        self.loss = loss
        return loss

    def backward(self, label_tensor):
        error_tensor = self.yk_hat
        error_tensor[label_tensor == 1] -= 1
        self.error_tensor = error_tensor
        return self.error_tensor
