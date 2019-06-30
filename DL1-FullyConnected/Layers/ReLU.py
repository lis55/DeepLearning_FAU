import numpy as np

class ReLU:
    def __init__(self):
        pass

    def forward(self, input_tensor):
        output_tensor = input_tensor * (input_tensor > 0) # takes the ones bigger than 0 
        self.output_tensor = output_tensor #save the output to be used in the backwards
        return output_tensor

    def backward(self, error_tensor):
        return error_tensor * (self.output_tensor > 0) #takes the ones larger than 0 (the positions of the elements from the previous input saved in the forward pass)
    
