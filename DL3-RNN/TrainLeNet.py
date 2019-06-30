from Layers import Helpers
import NeuralNetwork
import matplotlib.pyplot as plt
import os.path
import sys
from Models import LeNet


batch_size = 50
mnist = Helpers.MNISTData(batch_size)
mnist.show_random_training_image()
if os.path.isfile('trained/LeNet'):
    net = NeuralNetwork.load('trained/LeNet', mnist)
else:
    print('222')
    LeNet_class=LeNet.LeNet()
    LeNet_class.data_layer = mnist
    net = LeNet_class.build()
    net.data_layer = mnist
    print('net00',LeNet_class.build())

print('333')
print('net11',net)
net.train(10)
NeuralNetwork.save('trained/LeNet', net)
print('net.loss',type(net.loss))
plt.figure('Loss function for training LeNet on the MNIST dataset')
plt.plot(net.loss, '-x')
plt.show()
data, labels = net.data_layer.get_test_set() # return self.test, self.testLabels
results = net.test(data)
accuracy = Helpers.calculate_accuracy(results, labels)
print('\nOn the MNIST dataset, we achieve an accuracy of: ' + str(accuracy * 100) + '%')


