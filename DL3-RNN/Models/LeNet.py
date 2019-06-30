import numpy as np
import pickle
import NeuralNetwork
from Optimization import *
from Layers import *


class LeNet:

    def __init__(self):
        self.data_layer = None

    def build(self):
        input_tensor, label_tensor = self.data_layer.forward()
        print('input_tensor_build',input_tensor.shape)
        input_image_shape = (1, 28, 28)
        conv_stride_shape = (1, 1)
        num_kernels_1 = 6
        kernel_shape_1 = (1, 5, 5)
        num_kernels_2 = 16
        kernel_shape_2 = (6, 5, 5)
        pool_stride = (2, 2)
        pool_shape = (2, 2)
        optimizer = Optimizers.Adam(5e-4, 0.98, 0.999)
        optimizer.add_regularizer(Constraints.L2_Regularizer(4e-4))

        net = NeuralNetwork.NeuralNetwork(optimizer,
                                          Initializers.Xavier(),
                                          Initializers.Constant(0.1))

        net.loss_layer = SoftMax.SoftMax()
        conv_1 = Conv.Conv( conv_stride_shape, kernel_shape_1, num_kernels_1)
        net.layers.append(conv_1)
        pool_input_shape = list(np.divide(list(input_image_shape[1:-1]), list(conv_stride_shape)))
        pool_input_shape.insert(0, num_kernels_1)
        pool = Pooling.Pooling(pool_stride, pool_shape)
        net.layers.append(pool)
        dim3_pool_stride = list(pool_stride)
        dim3_pool_stride.insert(0, 1)
        conv_2_input_shape = tuple(np.divide(list(pool_input_shape), dim3_pool_stride))
        conv_2 = Conv.Conv( conv_stride_shape, kernel_shape_2, num_kernels_2)
        net.layers.append(conv_2)
        conv_2_out_shape=list(conv_2_input_shape)
        conv_2_out_shape[0]=num_kernels_2
        conv_2_out_shape=tuple(conv_2_out_shape)
        fully_batch_size=int(np.prod(conv_2_out_shape))
        net.layers.append(Flatten.Flatten())
        categories_1 = 84
        fcl_1 = FullyConnected.FullyConnected(fully_batch_size, categories_1)
        net.append_trainable_layer(fcl_1)
        net.layers.append(ReLU.ReLU())
        categories_2 = 10
        fcl_2 = FullyConnected.FullyConnected(categories_1, categories_2)
        net.append_trainable_layer(fcl_2)
        net.layers.append(ReLU.ReLU())
        net.loss_layer = SoftMax.SoftMax()

        return net
