import numpy as np


class SoftMax(object):  #

    loss = 0
    yk_hat = np.empty((0, 0))
    def __init__(self):
        pass

    def forward(self, input_tensor, label_tensor):
        self.input_tensor = input_tensor
        self.label_tensor = label_tensor
        self.yk_hat = SoftMax.predict(self, self.input_tensor)
        log_yk_hat = np.log(self.yk_hat)  # log_yk (9,4)
        np.place(log_yk_hat, label_tensor != 1, 0.0)  # save the needed loss where yk=1
        self.loss = np.sum(log_yk_hat) * -1.0
        return self.loss

    def predict(self, input_tensor):
        self.input_tensor = input_tensor
        maxvalue = np.max(self.input_tensor)
        xk = np.subtract(self.input_tensor, maxvalue)  # to increase numerical stability
        exp_x = np.exp(xk)
        self.yk_hat = np.divide(exp_x, np.expand_dims(np.sum(exp_x, axis=1), 1))
        return  self.yk_hat

    def backward(self, label_tensor):
        minus = np.zeros_like( self.yk_hat)
        np.place(minus, label_tensor == 1, 1.0)
        error_tensor =  self.yk_hat - minus
        return error_tensor
