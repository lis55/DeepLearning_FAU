
import numpy as np




# if (self.optimizer is not None):
#     self.weights_h = self.optimizer.calculate_update(self.delta, self.weights_h, self.gradient_weights_h)
#     self.weights_y = self.optimizer.calculate_update(self.delta, self.weights_y, self.gradient_weights_y)

a=np.random.uniform(low=0, high=1, size=(1,4))
b=np.random.uniform(low=0, high=1, size=(1,4))
# b=np.arange(0,9).reshape((3,3))
# # c=a[-1]
# # print(b.shape)
c=np.vstack((a,b ))
print(c.shape)
# for i in range (1,5):
#     print(i)

# t=np.delete(a, -1, axis=1)
# print('a',a,'t',t)