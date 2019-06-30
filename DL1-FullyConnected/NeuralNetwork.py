import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self):
        self.loss = []   #contain the loss value for each iteration after calling train
        self.layers = []  #arquitecture ?
        self.data_layer = None  #provides data and layers
        self.loss_layer = None  #softmax layer provides loss and prediction
        self.input_tensor = None
        self.label_tensor = None

    def forward(self):  #using input from the data_layer and passing it through all layers of the network
        input_tensor, self.label_tensor = self.data_layer.forward()    
        active_tensor=input_tensor
        #self.label_tensor = label_tensor
        for layer in self.layers:
            print('layer',layer)
            active_tensor=layer.forward(active_tensor)
        return self.loss_layer.forward(active_tensor,self.label_tensor)

    def backward(self):
        error_tensor = self.loss_layer.backward(self.label_tensor)    #Softmax
        for layer in reversed(self.layers):
            error_tensor = layer.backward(error_tensor)     #FullyConnected

    def train(self, iterations):  #train the weights (update in the backward)
        for i in range(0, iterations):
            self.loss.append(self.forward())
            self.backward()
    def test(self, input_tensor):   
        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)        
        prediction=self.loss_layer.predict(input_tensor)
        return prediction
