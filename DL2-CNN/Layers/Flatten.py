import numpy as np
class Flatten:
    def __init__(self):
        return None
    def forward(self,input_tensor):
        reshape=np.reshape(input_tensor,(np.shape(input_tensor)[0],-1)) #returns a 1D array
        self.inp=input_tensor #stores the shape of the input
        return reshape
    def backward(self,error_tensor):
        temp=np.ravel(error_tensor) #returns a 1D array
        reshape = np.reshape(temp,np.shape(self.inp)) #reshapes it again to the input size
        return reshape
