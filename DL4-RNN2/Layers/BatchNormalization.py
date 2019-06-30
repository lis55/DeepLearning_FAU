import numpy as np
import scipy as sc
from Layers import Base

class BatchNormalization:

    def __init__(self,*channels):
        self.channels = channels
        self.input_shape = (self.channels, 3, 3)
        self.epsilon =  1e-20
        self.mean_av=[]
        self.stand_var_av=[]
        self.phase=None

    def forward(self,input_tensor):
        self.input_tensor=input_tensor
        input_shape=self.input_tensor.shape
        alpha=0.8
        batch_size=self.input_tensor.shape[0]
        feature_size = np.prod(self.input_tensor.shape[1:])

        if (len(self.channels) !=0 ):
            self.new_size=(int(batch_size * feature_size / self.channels[0]), self.channels[0])
            new_tensor0=self.input_tensor
            new_tensor1=new_tensor0.reshape(input_shape[0],input_shape[1],input_shape[2]*input_shape[3])
            new_tensor2=np.transpose(new_tensor1,(0,2,1))
            self.new_tensor=new_tensor2.reshape((input_shape[0]*input_shape[2]*input_shape[3],input_shape[1]))
        else:
            self.new_size=(batch_size,feature_size)
            self.new_tensor = self.input_tensor

        mean_i=np.zeros((1,self.new_tensor.shape[1]))
        stand_var_i=np.zeros((1,self.new_tensor.shape[1]))
        if len(self.mean_av) == 0 and len(self.stand_var_av)==0:
            self.mean_av=mean_i
            self.stand_var_av=stand_var_i
            self.weights = np.ones((1, self.new_tensor.shape[1]))
            self.bias = np.zeros((1, self.new_tensor.shape[1]))

        for i in range( self.new_tensor.shape[1]):
            mean_i [0,i]= np.mean(self.new_tensor[:,i])  #
            stand_var_i [0,i]= np.sqrt(np.var(self.new_tensor[:,i]))
        if (self.phase is Base.Phase.train):
            self.mean_av = (1 - alpha) * self.mean_av + alpha * mean_i
            self.stand_var_av = (1 - alpha) * self.stand_var_av + alpha * stand_var_i

        if self.phase is Base.Phase.test:
            self.mean_use=self.mean_av
            self.stand_var_use=self.stand_var_av
        else:
            self.mean_use=mean_i
            self.stand_var_use = stand_var_i
        self.x_hat = self.new_tensor - self.mean_use
        self.x_hat = self.x_hat  / np.sqrt(self.stand_var_use**2+self.epsilon)
        self.y_hat = np.multiply(self.x_hat,self.weights) + self.bias
        if (len(self.channels) != 0):
            self.y_hatout = np.zeros_like(self.input_tensor)
            tt1=np.zeros((input_shape[0],input_shape[2]*input_shape[3],input_shape[1]))
            for h in range(self.channels [0]):
                tt0=self.y_hat[:, h].reshape(  # tt0 200,9
                    (self.input_tensor.shape[0], self.input_tensor.shape[2]*self.input_tensor.shape[3]))
                tt1[:, :, h] = tt0
            tt1=np.transpose(tt1,(0,2,1))
            tt2=tt1.reshape(input_shape)
            self.y_hatout=tt2
        else:
            self.y_hatout=self.y_hat

        return self.y_hatout

    def backward(self,error_tensor):
        self.error_input=error_tensor
        input_shape=self.error_input.shape
        if (len(self.channels) != 0):
            temp0 = self.error_input # 200 2 3 3
            print('temp0',temp0.shape)
            temp1 = temp0.reshape(input_shape[0], input_shape[1], input_shape[2]*input_shape[3])
            print('temp1',temp1.shape)
            temp2 = np.transpose(temp1, (0, 2, 1)) # 200 9 2
            print('temp2',temp2.shape)
            self.new_error = temp2.reshape(input_shape[0]*input_shape[2]*input_shape[3], input_shape[1])
            print('new_error',self.new_error.shape)
        else:
            self.new_error=self.error_input
        gradient_x_hat=self.new_error*self.weights
        print('gradient_x_hat',gradient_x_hat.shape)
        gradient_var = np.sum(gradient_x_hat * (self.new_tensor - self.mean_use) * -0.5 * (
                    (self.stand_var_use ** 2 + self.epsilon)**(-1.5)),axis=0)
        gradient_mu=np.sum(gradient_x_hat*(-1)/np.sqrt(self.stand_var_use**2+self.epsilon),axis=0)
        gradient_x = gradient_x_hat / np.sqrt(self.stand_var_use ** 2 + self.epsilon) + gradient_var * 2 * (
                    self.new_tensor - self.mean_use) / self.new_size[0]+gradient_mu/self.new_size[0]
        print('gradient_x',gradient_x.shape)
        if (len(self.channels) != 0):
            self.gradient_x_out = np.zeros_like(self.error_input)
            self.temp1 = np.zeros((input_shape[0], input_shape[2] * input_shape[3], input_shape[1]))
            for h in range(self.channels[0]):
                self.temp0=gradient_x[:, h].reshape((input_shape[0],input_shape[2]*input_shape[3])) # 1800--200 9 for 1 channel
                print('self.temp0',self.temp0.shape)
                self.temp1[:,:,h]=self.temp0
                print('self.temp1', self.temp1.shape)
            self.temp2=np.transpose(self.temp1,(0,2,1)) # 200 9 2 ---200 2 9
            print('self.temp2',self.temp2.shape)
            gradient_x=self.temp2.reshape(input_shape) # 200 2 9 ---200 2 3 3
            self.gradient_x_out=gradient_x
            print('gradient_x_out',self.gradient_x_out.shape)

        else:
            self.gradient_x_out = gradient_x
        self.get_gradient_weights()
        self.get_gradient_bias()

        return self.gradient_x_out

    def get_gradient_weights(self):
        gradient_weights = self.new_error*self.x_hat
        gradient_weights=np.sum(gradient_weights,axis=0)
        gradient_weights=np.expand_dims(gradient_weights,axis=0)

        return gradient_weights

    def get_gradient_bias(self):
        gradient_bias=np.sum(self.new_error,axis=0)
        gradient_bias=np.expand_dims(gradient_bias,axis=0)

        return gradient_bias

    def initialize(self,weights_initializer, bias_initializer):
        self.weights = weights_initializer

    def get_weights(self):

        return self.weights

    def set_weights(self, weights):
        self.weights = weights
