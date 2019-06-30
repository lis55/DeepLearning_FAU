import numpy as np
from scipy import signal

class Flatten():

    def __init__(self):
        pass

    def forward(self,input_tensor):
        for_input_tensor=input_tensor
        self.input_shape=list(for_input_tensor.shape)
        del self.input_shape[0]
        self.batch_size=for_input_tensor.shape[0]
        output_tensor = for_input_tensor.reshape(self.batch_size,np.prod(self.input_shape))

        return output_tensor

    def backward(self,input_tensor):
        back_input_tensor=input_tensor
        recover_shape=self.input_shape
        recover_shape.insert(0,self.batch_size)
        back_output_tensor=back_input_tensor.reshape(recover_shape)

        return back_output_tensor
