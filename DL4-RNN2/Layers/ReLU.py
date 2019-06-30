import numpy as np

import numpy as np

class ReLU():

    def __init__(self):
        pass

    def forward(self,input_tensor):
        self.input_tensor=input_tensor
        self.activated_tensor = np.maximum(0, self.input_tensor)

        return self.activated_tensor

    def backward(self,error_tensor):
        self.error_tensor=error_tensor
        np.place(self.error_tensor, self.activated_tensor <= 0.0, 0.0)

        return self.error_tensor


