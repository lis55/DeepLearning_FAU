import numpy as np

class TanH():

    def forward(self,input_tensor):
        self.fx_forward=np.tanh(input_tensor)
        return self.fx_forward

    def backward(self,error_tensor):
        self.fx_backward=(1-self.fx_forward**2)*error_tensor
        return self.fx_backward






