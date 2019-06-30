import numpy as np
from Layers import Base


class Dropout(Base.Base_class):
    def __init__(self,probability):
        super().__init__()
        self.probability=probability
        self.phase = Base.Phase.train

    def forward(self,input_tensor):
        self.input_tensor=input_tensor
        input_size=list(input_tensor.shape)
        self.dropout=np.ones_like(input_tensor)
        for i in range(input_size[0]):
            for j in range (input_size[1]):
                random_p=np.random.uniform(0, 1)
                if random_p >=self.probability:
                    self.dropout[i,j]=0

        if self.phase is Base.Phase.train:
            output_tensor = np.multiply(self.dropout, self.input_tensor)  # do dropout
            output_tensor=output_tensor/self.probability
        else:  # test
            output_tensor =input_tensor

        return output_tensor


    def backward(self,error_tensor):
        if self.phase is Base.Phase.train:
            error_tensor_out= np.multiply(self.dropout, error_tensor)
        else:
            error_tensor_out=error_tensor

        return error_tensor_out




