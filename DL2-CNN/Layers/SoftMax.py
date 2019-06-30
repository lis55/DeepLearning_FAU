import numpy as np


class SoftMax:

    def __init__(self):
        self.input_size = 0
        self.output_size = 0
        self.loss = 0
        self.input_tensor = None
        self.error_tensor = None
        self.y_hat = None

    def predict(self, input_tensor):
        current_input = input_tensor
        row_max = []
        sum = []
        for i in range(np.size(input_tensor, 0)):
            row_max.append(current_input[i, :].max())
        for i in range(np.size(input_tensor, 0)):
            for j in range(np.size(input_tensor, 1)):
                current_input[i][j] = current_input[i][j] - row_max[i]
        for i in range(np.size(input_tensor, 0)):
            for j in range(np.size(input_tensor, 1)):
                current_input[i][j] = np.exp(current_input[i][j])
        for i in range(np.size(input_tensor, 0)):
            s = 0.0
            for j in range(np.size(input_tensor, 1)):
                s += current_input[i][j]
            sum.append(s)
        for i in range(np.size(input_tensor, 0)):
            for j in range(np.size(input_tensor, 1)):
                current_input[i][j] = current_input[i][j] / sum[i]
        return current_input

    def forward(self, input_tensor, label_tensor):
        self.y_hat = self.predict(input_tensor)
        self.loss = 0.0
        for i in range(np.size(label_tensor, 0)):
            for j in range(np.size(label_tensor, 1)):
                if label_tensor[i][j] == 1:
                    self.loss -= np.log(self.y_hat[i][j])
        return self.loss

    def backward(self, label_tensor):
        self.error_tensor = self.y_hat
        for i in range(np.size(label_tensor, 0)):
            for j in range(np.size(label_tensor, 1)):
                if label_tensor[i][j] == 1:
                    self.error_tensor[i][j] -= 1
        return self.error_tensor