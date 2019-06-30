import numpy as np


class Pooling:
    def __init__(self, stride_shape, pooling_shape):

        self.input_image_shape = None
        self.stride_shape = stride_shape
        self.pooling_shape = pooling_shape
        self.output_tensor_shape = None
        self.error_tensor = None
        self.input_tensor = None

    def forward(self, input_tensor):
        self.input_image_shape = (
        input_tensor.shape[1], input_tensor.shape[2], input_tensor.shape[3])  ###input shape chanel,y,x
        self.input_tensor = input_tensor.reshape(input_tensor.shape[0], self.input_image_shape[0],
                                                 self.input_image_shape[1], self.input_image_shape[2])  # input tensor
        self.input_tensor = input_tensor
        size_first = int(np.floor(
            (self.input_tensor.shape[2] - self.pooling_shape[0]) / self.stride_shape[0]) + 1)  # size of the output
        size_second = int(np.floor((self.input_tensor.shape[3] - self.pooling_shape[1]) / self.stride_shape[1]) + 1)
        self.output_tensor = np.zeros((self.input_tensor.shape[0], self.input_tensor.shape[1], size_first, size_second))
        self.output_index = np.zeros((self.input_tensor.shape[0], self.input_tensor.shape[1], size_first,
                                      size_second))  # save the index for the backward
        for ba in range(self.input_tensor.shape[0]):  # pooling for every element of the batch and chanel
            for i in range(self.input_tensor.shape[1]):
                self.output_tensor[ba, i, :, :], self.output_index[ba, i, :, :] = self.pooling(
                    self.input_tensor[ba, i, :, :], self.pooling_shape[0], self.stride_shape[0], self.stride_shape[1],
                    'max') #function for the pooling defined bellow 

        self.output_tensor_shape = np.shape(self.output_tensor)
        self.output_tensor = self.output_tensor.reshape(self.output_tensor.shape[0],
                                                        self.output_tensor.shape[1],
                                                        self.output_tensor.shape[2],
                                                        self.output_tensor.shape[3])
        return self.output_tensor

    def backward(self, error_tensor):
        self.error_tensor = error_tensor.reshape(self.output_tensor_shape)
        error_tensor_next_layer = np.zeros(np.shape(self.input_tensor))
        for ba in range(self.output_tensor_shape[0]):
            for i in range(self.output_tensor_shape[1]):  # num of slides
                for j in range(self.output_tensor_shape[2]):
                    for k in range(self.output_tensor_shape[3]):
                        index_1 = int(np.floor(self.output_index[ba, i, j, k] / self.input_tensor.shape[3]))
                        index_2 = int(np.mod(self.output_index[ba, i, j, k], self.input_tensor.shape[3]))
                        error_tensor_next_layer[ba, i, index_1, index_2] += self.error_tensor[ba, i, j, k]
        error_tensor_next_layer = error_tensor_next_layer.reshape(error_tensor_next_layer.shape[0],
                                                                  error_tensor_next_layer.shape[1],
                                                                  error_tensor_next_layer.shape[2],
                                                                  error_tensor_next_layer.shape[3])
        return error_tensor_next_layer

    def pooling(self, inputMap, poolSize, poolStrideFirst, poolStrideSecond, mode='max'):
        in_row, in_col = np.shape(inputMap)  ##window
        out_row = in_row - poolSize + 1
        out_col = in_col - poolSize + 1
        outputMap = np.zeros((out_row, out_col))
        outputIndex = np.zeros((out_row, out_col))

        for r_idx in range(0, out_row):
            for c_idx in range(0, out_col):
                startY = r_idx
                startX = c_idx
                poolField = inputMap[startY:startY + poolSize, startX:startX + poolSize]
                poolOut = np.max(poolField)
                outputMap[r_idx, c_idx] = poolOut
                poolIndex = np.argmax(poolField) #saves the index of the maxima in the subarray
                devide_result = int(np.floor(poolIndex / poolSize))
                reminder_result = np.mod(poolIndex, poolSize) #element wise reminder of division
                real_row_index = startY + devide_result #repositions to the big array
                real_col_index = startX + reminder_result 
                outputIndex[r_idx, c_idx] = real_row_index * in_col + real_col_index

        out_row_stride = int(np.ceil(out_row / poolStrideFirst))
        out_col_stride = int(np.ceil(out_col / poolStrideSecond))
        outputMap_stride = np.zeros((out_row_stride, out_col_stride))
        outputIndex_stride = np.zeros((out_row_stride, out_col_stride)) #storing the value for the strided output

        for i in range(out_row_stride):
            for j in range(out_col_stride):
                outputMap_stride[i, j] = outputMap[i * poolStrideFirst, j * poolStrideSecond]
                outputIndex_stride[i, j] = outputIndex[i * poolStrideFirst, j * poolStrideSecond]

        return outputMap_stride, outputIndex_stride
