import numpy as np

class FullyConnected:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.uniform(0.0, 1.0, (output_size, input_size + 1))
        self.delta=1 ###this is supposed to be public ??self.__delta
        self.input_tensor = None
    def forward(self, input_tensor): ##input tensor is a matrix of arbitrary number of columns and batch_size number of rows
        self.input_tensor = np.hstack([input_tensor, np.ones([input_tensor.shape[0],1])]) 
        print(self.input_tensor.shape)
        output = np.dot(self.input_tensor, self.weights.T)
        
        return np.copy(output)
    
    def backward(self, error_tensor):
        current_weights = self.weights
        print(self.weights.shape)
        print(error_tensor.shape)
        #gradient with respect to x: dx=Y*WT/ En-1 = WT * En
        gradient_x = np.dot(error_tensor, current_weights)
 
        #gradient with respect to w
        self.gradient_weights = np.dot(error_tensor.T, self.input_tensor)
 
        #update weights using it's gradient
        #print(current_weights)
        self.weights = current_weights - self.delta * self.gradient_weights
 
        return np.copy(gradient_x[:, :-1])
    def get_gradient_weights(self):
        return np.copy(self.gradient_weights)
