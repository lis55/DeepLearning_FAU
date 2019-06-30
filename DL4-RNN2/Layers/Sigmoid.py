import numpy as np

class Sigmoid():

    def forward(self,input_tensor):
        self.fx_forward=1.0/(1.0+np.exp(-input_tensor))
        return self.fx_forward

    def backward(self,error_tensor):
        self.fx_backward=self.fx_forward*(1-self.fx_forward)*error_tensor
        return self.fx_backward