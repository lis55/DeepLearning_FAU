import numpy as np
from Layers import FullyConnected

class RNN():

    def __init__(self,input_size,hidden_size,output_size,bptt_length):
        self.input_size=input_size      # 13
        self.hidden_size=hidden_size    # 7
        self.output_size=output_size    # 5
        self.bptt_length=bptt_length    # 9
        self.fcl_h=FullyConnected.FullyConnected(hidden_size + input_size, hidden_size)
        self.fcl_y=FullyConnected.FullyConnected(hidden_size, output_size)
        self.hidden_state=None
        self.same_sequence=False
        self.last_iter_hidden_state=None
        self.optimizer=None
        self.delta = 1

    def toggle_memory(self):
        self.same_sequence=True

    def forward(self,input_tensor):
        self.input_tensor=input_tensor
        self.batch_size=self.input_tensor.shape[0]
        if self.same_sequence==False:
            self.hidden_state = np.zeros((self.batch_size+1,self.hidden_size))
        else:
            if self.hidden_state is None :
                self.hidden_state = np.zeros((self.batch_size + 1, self.hidden_size))
            else:
                self.hidden_state[0]=self.last_iter_hidden_state*self.same_sequence

        y_out=np.zeros((self.batch_size,self.output_size))

        for b in range (self.batch_size):
            new_input=np.hstack((np.expand_dims(self.hidden_state[b],0),np.expand_dims(self.input_tensor[b],0)))
            self.hidden_state[b+1]=np.tanh(self.fcl_h.forward(new_input))
            y_out[b]=(self.fcl_y.forward(np.expand_dims(self.hidden_state[b + 1],0)))

        self.last_iter_hidden_state=self.hidden_state[-1]
        return y_out

    def backward(self,error_tensor):
        print('error_tensor',error_tensor.shape)
        self.error_tensor=error_tensor
        self.error_tensor_out=np.zeros((self.batch_size,self.input_size))
        hx_size=self.hidden_size + self.input_size
        steps=1
        self.gradient_weights_y = np.zeros((self.hidden_size+1, self.output_size))
        self.gradient_weights_hx = np.zeros((hx_size+1, self.hidden_size))

        gradient_tanh=1 - self.hidden_state[1::] ** 2
        error_h=np.zeros((1,self.hidden_size))  # backward

        for b in reversed(range(self.batch_size)):
            one_batch_error=self.error_tensor[b]
            error_y_h = self.fcl_y.backward(np.expand_dims(one_batch_error,0))
            self.fcl_y.input_tensor=np.expand_dims(np.hstack((self.hidden_state[b+1],1)),0)

            gra_y_ht=error_h+error_y_h
            print('ht,gradient_tanh',error_y_h.shape,error_h.shape,gra_y_ht.shape,gradient_tanh[b].shape)
            gradient_hidden_t=gradient_tanh[b]*gra_y_ht
            error_hx = self.fcl_h.backward(gradient_hidden_t)
            error_h = error_hx[:, 0:self.hidden_size]   # hidden
            error_x = error_hx[:, self.hidden_size:hx_size + 1]
            self.error_tensor_out[b]=error_x
            self.fcl_h.input_tensor=np.expand_dims(np.hstack((self.hidden_state[b],self.input_tensor[b],1)),0)

            steps+=1
            if steps<=self.bptt_length:
                self.weights_y=self.fcl_y.get_weights()
                self.weights_h=self.fcl_h.get_weights()
                self.get_gradient_weights()

        if self.optimizer is not None:
            self.weights_y = self.optimizer.calculate_update(self.delta, self.weights_y, self.gradient_weights_y)
            self.weights_h = self.optimizer.calculate_update(self.delta, self.weights_h, self.gradient_weights_hx)
            self.fcl_y.set_weights(self.weights_y)
            self.fcl_h.set_weights(self.weights_h)

        return self.error_tensor_out

    def get_gradient_weights(self):
       self.gradient_weights_y+=self.fcl_y.get_gradient_weights()
       self.gradient_weights_hx+=self.fcl_h.get_gradient_weights()
       return self.gradient_weights_hx

    def get_weights(self):
        weights=self.fcl_h.get_weights()
        return weights

    def set_weights(self,weights):
        self.fcl_h.set_weights(weights)

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def initialize(self, weights_initializer, bias_initializer):
        self.fcl_y.initialize(weights_initializer, bias_initializer)
        self.fcl_h.initialize(weights_initializer, bias_initializer)

