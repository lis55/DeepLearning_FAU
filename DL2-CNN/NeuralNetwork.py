import numpy as np
import matplotlib.pyplot as plt
import copy

'''
class NeuralNetwork:
    #def __init__(self,uniform_ini,constant_ini,weights_initializer,bias_initializer):
    def __init__(self,optimizer,weights_initializer,bias_initializer):
        self.loss = []   #contain the loss value for each iteration after calling train
        self.layers = []  #arquitecture ?
        self.data_layer = None  
        self.loss_layer = None  
        self.input_tensor = None
        self.label_tensor = None
        #self.uni=Initializers.UniformRandom()
        #self.const=Initializers.Constant(0.1)
        self.weights_initializer = weights_initializer
        self.bias_initializer = bias_initializer
        self.optimizer = optimizer

    def forward(self):  #using input from the data_layer and passing it through all layers of the network
        input_tensor, self.label_tensor = self.data_layer.forward()    
        active_tensor=input_tensor
        #self.label_tensor = label_tensor
        for layer in self.layers:
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

    def append_trainable_layer(self,layer):
        layer.initialize=(self.weights_initializer,self.bias_initializer)
        layer.set_optimizer(self.optimizer)
        self.layers.append_trainable_layer(layer)

    def append_trainable_layer(self,layer):
        layer.initialize(copy.deepcopy(self.weights_initializer), copy.deepcopy(self.bias_initializer))
        layer.set_optimizer(copy.deepcopy(self.optimizer))
        self.layers.append(layer)


'''
class NeuralNetwork:

    def __init__(self, optimizer, weights_initializer, bias_initializer, ):
        self.loss = []
        self.layers = []
        self.data_layer = None
        self.loss_layer = None
        self.label_tensor = None
        self.weights_initializer = weights_initializer
        self.bias_initializer = bias_initializer
        self.optimizer = optimizer

    def forward(self):
        input_tensor, self.label_tensor = self.data_layer.forward()
        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)
        loss = self.loss_layer.forward(input_tensor, self.label_tensor)
        return loss

    def backward(self):
        error_tensor = self.loss_layer.backward(self.label_tensor)
        for layer in reversed(self.layers):
            error_tensor = layer.backward(error_tensor)

    def train(self, iterations):
        for i in range(iterations):
            self.forward()
            self.backward()
            self.loss.append(self.loss_layer.loss)
        return self.loss

    def test(self, input_tensor):
        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)
        return self.loss_layer.predict(input_tensor)

    def append_trainable_layer(self, layer):
        layer.initialize(copy.deepcopy(self.weights_initializer), copy.deepcopy(self.bias_initializer))
        layer.set_optimizer(copy.deepcopy(self.optimizer))
        self.layers.append(layer)
        
        
