import numpy as np
from Optimization import *
import math


class Pooling:
    def __init__(self, stride_shape, pooling_shape):
        self.stride_shape = stride_shape
        self.pooling_shape = pooling_shape

    def forward(self, input_tensor):  # input_tensor 2*(2*4*7)=2*56 random
        self.batch_size = input_tensor.shape[0]
        self.input_image_shape = input_tensor.shape[1:]
        self.newsize = (
            self.batch_size, self.input_image_shape[0], self.input_image_shape[1], self.input_image_shape[2])
        self.input_tensor = input_tensor.reshape(self.newsize)
        self.recover_tensor = np.zeros(self.newsize)

        ty = 0
        tx = 0
        for j in range(0, self.input_image_shape[1], self.stride_shape[0]):
            if j + self.pooling_shape[0] > self.input_image_shape[1]:
                break
            ty = ty + 1
        for k in range(0, self.input_image_shape[2], self.stride_shape[1]):
            if k + self.pooling_shape[1] > self.input_image_shape[2]:
                break
            tx = tx + 1

        output_tensor = np.zeros((self.batch_size, self.input_image_shape[0], ty, tx))
        max_position_size = output_tensor.shape
        max_position_size = list(max_position_size)
        max_position_size.insert(4, 2)
        self.max_position_size = tuple(max_position_size)
        self.max_position = np.zeros(self.max_position_size)

        for b in range(self.batch_size):
            for i in range(self.input_image_shape[0]):  # z
                newy = 0
                for j in range(0, self.input_image_shape[1], self.stride_shape[0]):  # y -- y_pool
                    if j + self.pooling_shape[0] > self.input_image_shape[1]:
                        break
                    newx = 0
                    for k in range(0, self.input_image_shape[2], self.stride_shape[1]):  # x -- x_pool
                        if k + self.pooling_shape[1] > self.input_image_shape[2]:
                            break
                        pool_b = self.input_tensor[b, i, j:j + self.pooling_shape[0], k:k + self.pooling_shape[1]]
                        output_tensor[b, i, newy, newx] = np.max(pool_b)
                        order = np.argmax(pool_b, axis=None)
                        row = order // self.pooling_shape[0]
                        column = order % self.pooling_shape[0]
                        self.recover_tensor[b, i, j + row, k + column] = self.input_tensor[b, i, j + row, k + column]
                        self.max_position[b, i, newy, newx, 0] = j + row  # y
                        self.max_position[b, i, newy, newx, 1] = k + column  # x
                        newx += 1
                    newy += 1
        self.downsample_size = output_tensor.shape
        return output_tensor  # last result need to reshape

    def backward(self, error_tensor):
        self.error_tensor = error_tensor
        self.error_tensor=self.error_tensor.reshape(self.downsample_size)
        upsample_size=self.newsize
        self.pool_upsample = np.zeros(upsample_size)

        for b in range(self.batch_size):
            for i in range(self.input_image_shape[0]):
                for j in range(self.max_position_size[2]):
                    for k in range(self.max_position_size[3]):
                        ori_y = self.max_position[b, i, j, k, 0]
                        ori_x = self.max_position[b, i, j, k, 1]
                        y = int(ori_y)
                        x = int(ori_x)
                        self.pool_upsample[b, i, y, x] += self.error_tensor[b, i, j, k]

        return self.pool_upsample
