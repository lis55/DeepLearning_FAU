import copy
import scipy
from scipy import signal
import numpy as np


class Conv:
    def __init__(self, stride_shape, convolution_shape, num_kernels, delta=1):
        self.stride_shape = stride_shape
        self.convolution_shape = convolution_shape
        self.num_kernels = num_kernels
        self.delta = delta
        self.input_shape = None
        self.output_shape = None
        self.error_tensor_upsamp = None

        if len(convolution_shape) == 3: #for 3d convolution
            self.weights = np.random.rand(num_kernels, convolution_shape[0], convolution_shape[1], convolution_shape[2])
            self.bias = np.random.rand(num_kernels) #1 bias per kernel
            self.stride_row = self.stride_shape[0]
            self.stride_col = self.stride_shape[1]
            self.convolution_row_shape = convolution_shape[1]
            self.convolution_col_shape = convolution_shape[2]
            self.dim1 = False
        else: #distinction for 2d convlution
            self.weights = np.random.rand(num_kernels, convolution_shape[0], convolution_shape[1], 1)
            self.bias = np.random.rand(num_kernels)
            self.stride_row = self.stride_shape[0]
            self.stride_col = 1
            self.convolution_row_shape = convolution_shape[1]
            self.convolution_col_shape = 1
            self.dim1 = True #boolean for the 2d case

        self.input_tensor = None
        self.error_tensor = None
        self.gradient_weights = None
        self.gradient_bias = None
        self.weightsOptimizer = None
        self.biasOptimizer = None

    def forward(self, input_tensor):  # input dimensions batch,c,y,x
        # np.shape(input_tensor)
        ###We distinguish between 2d and 3d case
        if self.dim1:
            self.input_shape = (input_tensor.shape[1], input_tensor.shape[2], 1)  ###chanels y,1
            self.input_tensor = input_tensor.reshape(input_tensor.shape[0], self.input_shape[0], self.input_shape[1],
                                                     1)  ###reshape input tensor batch, chanels, y,1

        else:
            self.input_shape = (input_tensor.shape[1], input_tensor.shape[2], input_tensor.shape[3])  ##chanels, y, x
            self.input_tensor = input_tensor.reshape(input_tensor.shape[0], self.input_shape[0], self.input_shape[1],
                                                     self.input_shape[2])  ###reshape input tensor batch, chanels, y,x

        ##set the output shape
        output_tensor = np.zeros(
            (input_tensor.shape[0], self.num_kernels, self.input_tensor.shape[2],
             self.input_tensor.shape[3]))  # output tensor batch,chanels=num_kernels,y,x
        for ba in range(input_tensor.shape[0]):  # batches
            for i in range(self.num_kernels):  # kernels (chanels of the output)
                for j in range(self.input_tensor.shape[1]):  # chanels of the input
                    output_tensor[ba, i, :, :] += scipy.signal.correlate2d(self.input_tensor[ba, j, :, :],
                                                                           self.weights[i, j, :, :], 'same') # we correlate each channel 
        # add the bias to each element
        for ba in range(input_tensor.shape[0]):
            for num in range(self.num_kernels):
                for i in range(output_tensor.shape[2]):
                    for j in range(output_tensor.shape[3]):
                        output_tensor[ba, num, i, j] += self.bias[num]
        ###we stride
        size_first = int(np.ceil(output_tensor.shape[2] / self.stride_row))
        size_second = int(np.ceil(output_tensor.shape[3] / self.stride_col))
        output_tensor_with_stride = np.zeros((input_tensor.shape[0], self.num_kernels, size_first, size_second))
        for ba in range(input_tensor.shape[0]):
            for i in range(self.num_kernels):
                for j in range(size_first):
                    for k in range(size_second):
                        j_in_output_tensor = j * self.stride_row
                        k_in_output_tensor = k * self.stride_col
                        output_tensor_with_stride[ba, i, j, k] = output_tensor[
                            ba, i, j_in_output_tensor, k_in_output_tensor]
        self.output_shape = np.shape(output_tensor_with_stride)  # store the output shape

        #again distinction between 2d and 3d
        if self.dim1:
            output_tensor_with_stride = output_tensor_with_stride.reshape(output_tensor_with_stride.shape[0],
                                                                          output_tensor_with_stride.shape[1],
                                                                          output_tensor_with_stride.shape[2])
        else:
            output_tensor_with_stride = output_tensor_with_stride.reshape(output_tensor_with_stride.shape[0],
                                                                          output_tensor_with_stride.shape[1],
                                                                          output_tensor_with_stride.shape[2],
                                                                          output_tensor_with_stride.shape[3])
        return output_tensor_with_stride

    def backward(self, error_tensor):
        self.error_tensor = error_tensor.reshape(self.output_shape)  ##error tensor has the dimensions of the output saved in the forward
        # upsampling
        self.error_tensor_upsamp = np.zeros((self.input_tensor.shape[0], self.num_kernels, self.input_tensor.shape[2],
                                             self.input_tensor.shape[3]))  # num_ker=num chanels
        for ba in range(self.error_tensor_upsamp.shape[0]):
            for num in range(self.error_tensor_upsamp.shape[1]):
                for i in range(self.error_tensor.shape[2]):
                    for j in range(self.error_tensor.shape[3]):
                        self.error_tensor_upsamp[ba, num, i * self.stride_row, j * self.stride_col] = self.error_tensor[
                            ba, num, i, j]  # we fill up with the strided error tensor

        error_tensor_next_layer = np.zeros(np.shape(self.input_tensor))  # we have the same size of the input
        for ba in range(self.error_tensor.shape[0]):  # batch num
            for i in range(self.input_tensor.shape[1]):  # channel num
                for j in range(self.num_kernels):
                    temp = scipy.signal.convolve2d(self.error_tensor_upsamp[ba, j, :, :], self.weights[j, i, :, :],
                                                   'same')  # same
                    error_tensor_next_layer[ba, i, :, :] += temp #we convolved every channel of the upsampled error tensor with every channel of the weights

        # input padding(right) we pad with half of the kernel size
        up_size = int(np.floor(self.convolution_col_shape / 2))  # (3, 5, 8)
        down_size = self.convolution_col_shape - up_size - 1
        left_size = int(np.floor(self.convolution_row_shape / 2))
        right_size = self.convolution_row_shape - left_size - 1

        self.input_tensor_padding = np.zeros((self.input_tensor.shape[0], self.input_tensor.shape[1],
                                              self.input_tensor.shape[2] + self.convolution_row_shape - 1,
                                              self.input_tensor.shape[3] + self.convolution_col_shape - 1))
        for ba in range(self.input_tensor.shape[0]):
            for num in range(self.input_tensor.shape[1]):
                for i in range(self.input_tensor_padding.shape[2]):
                    for j in range(self.input_tensor_padding.shape[3]):
                        if (i > left_size - 1) and (i < self.input_tensor.shape[2] + left_size):
                            if (j > up_size - 1) and (j < self.input_tensor.shape[3] + up_size):
                                temp = self.input_tensor[ba, num, i - left_size, j - up_size]
                                self.input_tensor_padding[ba, num, i, j] = temp
        for ba in range(self.error_tensor_upsamp.shape[0]):
            for i in range(self.error_tensor_upsamp.shape[1]):
                self.error_tensor_upsamp[ba, i, :, :] = self.FZ(self.error_tensor_upsamp[ba, i, :, :])  #FZ fucntion defined at the end

        #gradient with respect to the weights
        self.gradient_weights = np.zeros(
            (self.weights.shape[0], self.weights.shape[1], self.weights.shape[2], self.weights.shape[3]))
        for ba in range(self.error_tensor.shape[0]):
            for i in range(self.num_kernels):
                for j in range(self.input_tensor.shape[1]):
                    self.gradient_weights[i, j, :, :] += scipy.signal.convolve2d(self.input_tensor_padding[ba, j, :, :],
                                                                                 self.error_tensor_upsamp[ba, i, :, :],
                                                                                 'valid') #convolution of the error tensor with the padded input tensor
        # gradient with respect to the bias
        self.gradient_bias = np.zeros(self.num_kernels)
        gradient_bias_mid = np.zeros((self.error_tensor.shape[0], self.error_tensor.shape[1]))
        for ba in range(self.error_tensor.shape[0]):
            for i in range(self.error_tensor.shape[1]):  # bias
                gradient_bias_mid[ba, i] = np.sum(self.error_tensor[ba, i, :, :])

        temp = np.sum(gradient_bias_mid, 0)
        for i in range(self.error_tensor.shape[1]):
            self.gradient_bias[i] = temp[i]
        if self.weightsOptimizer is not None:
            self.weights = self.weightsOptimizer.calculate_update( self.weights, self.get_gradient_weights())

        self.b1 = self.bias
        self.b = self.get_gradient_bias()
        if self.biasOptimizer is not None:
            self.bias = self.biasOptimizer.calculate_update(self.bias, self.get_gradient_bias())
        
        #again distinction between 2d and 3d
        if self.dim1:
            error_tensor_next_layer = error_tensor_next_layer.reshape(error_tensor_next_layer.shape[0],
                                                                      error_tensor_next_layer.shape[1],
                                                                      error_tensor_next_layer.shape[2])
        else:
            error_tensor_next_layer = error_tensor_next_layer.reshape(error_tensor_next_layer.shape[0],
                                                                      error_tensor_next_layer.shape[1],
                                                                      error_tensor_next_layer.shape[2],
                                                                      error_tensor_next_layer.shape[3])
        return error_tensor_next_layer

    def set_optimizer(self, optimizer):
        self.weightsOptimizer = copy.deepcopy(optimizer)
        self.biasOptimizer = copy.deepcopy(optimizer)
        return

    def get_gradient_weights(self):
        return self.gradient_weights

    def get_gradient_bias(self):
        return self.gradient_bias

    def initialize(self, weights_initializer, bias_initializer):
        self.weights = weights_initializer.initialize(np.shape(self.weights), np.prod(self.convolution_shape),
                                                      np.prod(self.convolution_shape[1:]) * self.num_kernels)
        self.bias = bias_initializer.initialize(np.shape(self.bias), 1, np.shape(self.weights)[0])
        return

    def fz(self, a): ##function that reverses the order of elements
        return a[::-1]

    def FZ(self, mat):
        return np.array(self.fz(list(map(self.fz, mat))))  #reverses the order of every row of the array and the order of rows

