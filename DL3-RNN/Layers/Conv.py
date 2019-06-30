import numpy as np
from scipy import signal
from copy import *
import math
import cv2
from Optimization import *


class Conv:

    weights=None
    bias=None
    optimizer=None

    def __init__(self,  stride_shape, kernel_shape, number_kernels,*parameter):
        self.stride_shape = stride_shape
        self.kernel_shape = kernel_shape  #(3, 5, 8)
        self.number_kernels = number_kernels
        kernels_shape=list(self.kernel_shape)
        kernels_shape.insert(0,self.number_kernels)
        kernels_shape=tuple(kernels_shape)
        print('kernels_shape',kernels_shape)

        self.weights = np.random.uniform(low=0, high=1, size=kernels_shape)
        self.bias = np.random.uniform(0, 1, (self.number_kernels ))

        if len(parameter)==1:
            self.delta=parameter[0]

        else:
            self.delta = 1


    def forward(self, input_tensor):
        self.batch_size = input_tensor.shape[0]
        self.img_shape=input_tensor.shape[1:]
        self.reshape_img_size=list(self.img_shape)
        self.reshape_img_size.insert(0,self.batch_size)
        self.reshape_img_size=tuple(self.reshape_img_size)
        self.input_tensor = np.array(input_tensor)  # input(batch,(3, 10, 14))  kernel(3, 5, 8)
        self.input_tensor = self.input_tensor.reshape(self.reshape_img_size)  # reshape as (b,z,y,x)
        biasmat_size=list(self.img_shape)
        del biasmat_size[0]
        biasmat_size.insert(0, self.number_kernels)
        biasmat_size.insert(0,self.batch_size)
        self.biasmat_size=biasmat_size
        self.biasMat = np.ones(biasmat_size)
        self.biasMat=np.ones(self.biasmat_size) # (b,k,y,x)

        for b in range(self.biasmat_size[0]):
            for k in range(self.number_kernels):
                self.biasMat[b,k] = self.biasMat[b,k] * self.bias[k]

        add_input_size=list(self.img_shape)
        del add_input_size[0]
        add_input_size.insert(0,self.batch_size)
        add_input_size=tuple(add_input_size)
        self.add_input=np.ones(add_input_size)
        self.output_size=self.reshape_img_size
        self.output_size=list(self.output_size)
        self.output_size[1]=self.number_kernels  # number z change to self.number_kernels
        self.output_size=tuple(self.output_size)
        self.output_tensor = np.zeros(self.output_size)

        get_input_size = np.zeros(self.img_shape)  # be used to judge the input dimension

        if not self.weights is None:
            self.weights=self.weights

        if get_input_size.ndim==3:
            sub_y = self.img_shape[1] // self.stride_shape[0]
            if self.img_shape[1] % self.stride_shape[0] != 0:  # if rest still has space it can still conv see from test file
                sub_y += 1
            sub_x = self.img_shape[2] // self.stride_shape[1]
            if self.img_shape[2] % self.stride_shape[1] != 0:
                sub_x += 1
            self.sub_output_tensor = np.zeros((self.batch_size, self.number_kernels, sub_y, sub_x))
        else:
            sub_y = self.img_shape[1] // self.stride_shape[0]
            if self.img_shape[1] % self.stride_shape[0] != 0:  # if rest still has space it can still conv see from test file
                sub_y += 1
            self.sub_output_tensor = np.zeros((self.batch_size, self.number_kernels, sub_y))

        for b in range(self.batch_size):  # convolution  first
            for k in range(self.number_kernels):   # one kernel get one layer
                    kernel = self.weights[k]
                    temp = signal.correlate(self.input_tensor[b], kernel, mode='same')
                    self.output_tensor[b, k] = temp[math.floor(self.img_shape[0] / 2)]
                    self.output_tensor[b, k] += self.biasMat[b,k]

        for b in range(self.batch_size):  # and then downsampling
            for i in range(sub_y):  # y  img:10  kernel:5
                if  get_input_size.ndim==3:
                    for j in range(sub_x):
                        self.sub_output_tensor[b, :, i, j] = self.output_tensor[b, :, i * self.stride_shape[0],
                                                             j * self.stride_shape[1]]
                else:
                    self.sub_output_tensor[b, :, i] = self.output_tensor[b, :, i * self.stride_shape[0]]

        sub_out_shape = self.sub_output_tensor.shape
        sub_out_reshape=list(sub_out_shape)
        del sub_out_reshape[0]
        sub_out_reshape=tuple(sub_out_reshape)
        return_shape = np.prod(sub_out_reshape)

        return self.sub_output_tensor

    def backward(self, error_tensor):
        self.error_tensor = np.array(error_tensor)
        get_input_size=np.zeros(self.img_shape)
        reshape_size=None
        if get_input_size.ndim==3:
            sub_y = self.img_shape[1] // self.stride_shape[0]
            if self.img_shape[1] % self.stride_shape[0] != 0:
                sub_y += 1
            sub_x = self.img_shape[2] // self.stride_shape[1]
            if self.img_shape[2] % self.stride_shape[1] != 0:
                sub_x += 1
            reshape_size = (self.batch_size, self.number_kernels, sub_y, sub_x)
            upsamlpe_size=(self.batch_size, self.number_kernels, self.img_shape[1], self.img_shape[2])
            ori_yx_size = (self.img_shape[2], self.img_shape[1])
        else:
            sub_y = self.img_shape[1] // self.stride_shape[0]
            if self.img_shape[1] % self.stride_shape[0] != 0:
                sub_y += 1
                reshape_size = (self.batch_size, self.number_kernels, sub_y)
            upsamlpe_size=(self.batch_size,self.number_kernels,self.img_shape[1])
            ori_zy_size = (self.img_shape[1],self.number_kernels)

        self.error_tensor = self.error_tensor.reshape(reshape_size)  # (b,z,y,x)
        self.error_upsample_tensor = np.zeros(upsamlpe_size)

        if get_input_size.ndim == 3:
            for i in range(self.error_tensor.shape[2]):
                for j in range(self.error_tensor.shape[3]):
                    self.error_upsample_tensor[:, :,i*self.stride_shape[0],j*self.stride_shape[1]]=self.error_tensor[:,:,i,j]

        else:  #z y resize directly
            for i in range(self.error_tensor.shape[2]):
                self.error_upsample_tensor[:, :, i*self.stride_shape[0]] =self.error_tensor[:,:,i]

        new_kernels_size=list(self.kernel_shape)
        new_kernels_size.insert(1,self.number_kernels)
        new_kernels_size=tuple(new_kernels_size)
        new_kernels=np.zeros(new_kernels_size)
        for i in range(self.kernel_shape[0]): #kernel_shape[0]=3
            for j in range(self.number_kernels):  #self.number_kernels=4
                if get_input_size.ndim == 3:
                    new_kernels[i, j, :, :] = self.weights[j, i, :, :]
                else:
                    new_kernels[i, j, :] = self.weights[j, i, :]

        error_out_size=list(self.img_shape)
        error_out_size.insert(0,self.batch_size)
        error_out_size=tuple(error_out_size)
        error_con_out_tensor = np.zeros(error_out_size)

        for b in range(self.batch_size):
            for i in range(new_kernels.shape[0]):
                kernel = new_kernels[i]
                kernel = kernel[::-1]
                extract=math.floor(self.error_upsample_tensor.shape[1] / 2)
                if get_input_size.ndim == 3:
                    error_con_out_tensor[b, i, :, :] = signal.convolve(self.error_upsample_tensor[b, :, :, :], kernel,
                                                                       mode='same')[extract,:, :]
                else:
                    error_con_out_tensor[b, i, :] = signal.convolve(self.error_upsample_tensor[b, :, :], kernel,
                                                                       mode='same')[extract, :]

        if not self.optimizer is None:
            self.get_gradient_weights()
            self.get_gradient_bias()
            self.weights = self.optimizer.calculate_update(1, self.weights, self.gradient_weight_out)
            self.bias = self.optimizer.calculate_update(1, self.bias,  self.gradient_bias)


        error_out_tensor = error_con_out_tensor.reshape(self.batch_size,np.prod(self.img_shape))
        return error_con_out_tensor  # (batch,z,y,x)

    def set_optimizer(self, optimizer):
        self.optimizer=deepcopy(optimizer)

    def initialize(self, weight_initializer, bias_initializer):
        fan_in=self.weights.shape[1] * self.weights.shape[2] * self.weights.shape[3]
        fan_out=self.number_kernels * self.weights.shape[2] * self.weights.shape[3]
        self.weights = weight_initializer.initialize(self.weights,fan_in,fan_out) #use TestInitializer.initialize
        self.bias = bias_initializer.initialize(self.bias,None,None)

        return self.weights, self.bias

    def get_gradient_weights(self):
        get_input_size=np.zeros(self.img_shape)
        if get_input_size.ndim==3:
            ky = self.kernel_shape[1]
            kx = self.kernel_shape[2]
            pad_u = math.ceil((ky - 1) / 2.0)
            pad_d = math.floor((ky - 1) / 2.0)
            pad_l = math.ceil((kx - 1) / 2.0)
            pad_r = math.floor((kx - 1) / 2.0)
            self.pad_input_tensor = np.zeros(
                (self.batch_size, self.img_shape[0], self.img_shape[1] + ky - 1, self.img_shape[2] + kx - 1))
            self.pad_input_tensor[:, :, pad_u:-pad_d, pad_l:-pad_r] = self.input_tensor
        else:
            ky = self.kernel_shape[1]
            pad_u = math.ceil((ky - 1) / 2.0)
            pad_d = math.floor((ky - 1) / 2.0)
            self.pad_input_tensor = np.zeros(
                (self.batch_size, self.img_shape[0], self.img_shape[1] + ky - 1))
            self.pad_input_tensor[:, :, pad_u:-pad_d] = self.input_tensor

        gradient_weight_out_size=list(self.kernel_shape)
        gradient_weight_out_size.insert(0,self.number_kernels)
        gradient_weight_out_size=tuple(gradient_weight_out_size)  #(b,z,y,x)
        self.gradient_weight_out=np.zeros(gradient_weight_out_size)
        temp_weights=np.zeros_like(self.gradient_weight_out)

        for b in range(self.batch_size):
            for i in range(self.number_kernels):
                if get_input_size.ndim == 3:
                    temp = signal.correlate(self.pad_input_tensor[b, :, :, :],
                                           np.expand_dims(self.error_upsample_tensor[b, i, :, :], axis=0),
                                           mode='valid')
                    temp_weights[i]=temp

                else:
                    temp = signal.correlate(self.pad_input_tensor[b, :, :],
                                           np.expand_dims(self.error_upsample_tensor[b, i, :], axis=0),
                                           mode='valid')
                    temp_weights[i] = temp

            self.gradient_weight_out=self.gradient_weight_out+ temp_weights

        return self.gradient_weight_out

    def get_gradient_bias(self):
        self.gradient_bias = np.zeros_like(self.bias)
        for i in range(self.number_kernels):
            for b in range(self.error_tensor.shape[0]):
                self.gradient_bias[i] += np.sum(self.error_tensor[b, i])
        return self.gradient_bias
