# import numpy as np
# import pickle
# import numpy as np
# import matplotlib.pyplot as plt
# import os
# import gzip
# import struct
# from pathlib import Path
# from random import shuffle
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.datasets import load_iris, load_digits
#
# # a = np.arange(54).reshape(3, 18)
# # # print(a[1:3])
# #
# #
# # aa = np.ones((1, 18))
# # rr = np.multiply(aa, a)
#
# # rr2 = a ** -1.5
# # rr3 = np.power(a, -1.5)
# # print(rr2)
# # print(rr3)
#
# # b = np.ones((1, 18))
#
# # class addition:
# #
# #
# #     def add(s, *arg):
# #         if arg != None:
# #             print(arg)
# #         if len(arg) == 3:
# #             res = arg[2]
# #             print(res)
#
# # summer = add(1)  # construct an object
#
# # picklestring = pickle.dumps(summer)  # serialize object
# #
# # a = {'hello': 'world'}
# #
# # fn='filename.pickle'
# # with open(fn, 'wb') as handle:
# #     pickle.dump(addition(), handle)
# #
# # with open('filename.pickle', 'rb') as handle:
# #     b = pickle.load(handle)
# #
# #
# # print(b)
#
#
# def _read(dataset="training"):
#     """
#     Python function for importing the MNIST data set.  It returns an iterator
#     of 2-tuples with the first element being the label and the second element
#     being a numpy.uint8 2D array of pixel data for the given image.
#     """
#     print('111')
#     root_dir = Path(__file__)
#
#     if dataset is "training":
#         fname_img = root_dir.parent.parent.joinpath('Data', 'train-images-idx3-ubyte.gz')
#         fname_lbl = root_dir.parent.parent.joinpath('Data', 'train-labels-idx1-ubyte.gz')
#     elif dataset is "testing":
#         fname_img = root_dir.parent.parent.joinpath('Data', 't10k-images-idx3-ubyte.gz')
#         fname_lbl = root_dir.parent.parent.joinpath('Data', 't10k-labels-idx1-ubyte.gz')
#     else:
#         raise ValueError("dataset must be 'testing' or 'training'")
#
#     # Load everything in some numpy arrays
#     with gzip.open(str(fname_lbl), 'rb') as flbl:
#         magic, num = struct.unpack(">II", flbl.read(8))
#
#         s = flbl.read(num)
#         lbl = np.frombuffer(s, dtype=np.int8)
#         one_hot = np.zeros((lbl.shape[0], 10))
#         for idx, l in enumerate(lbl):
#             one_hot[idx, l] = 1
#
#     with gzip.open(str(fname_img), 'rb') as fimg:
#         magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
#
#         buffer = fimg.read(num * 32 * 32 * 8)
#         img = np.frombuffer(buffer, dtype=np.uint8).reshape(len(lbl), rows * cols)
#         img = img.astype(np.float64)
#         img /= 255.0
#
#     img = img[:num, :]
#
#     # print('img',sum(img))
#     tt=img[0]
#     tt=tt.reshape((rows , cols))
#     print('img',tt.shape)
#     plt.figure()
#     plt.imshow(tt)
#     plt.show()
#
#     return img
#
# # a=np.arange(5)
# # c=np.random.choice(a,5)
#
# # print(c,a)

# a=(6,28,28)
# b=(5,5)
# c=(1,1)
# pool=(2,2)
#
# # res=np.divide(list(a),list(c))-list(b)+1
# # temp=np.expand_dims(list(pool),axis=0)
# temp=list(pool)
# temp.insert(0,1)
# pres=np.divide(list(a),temp)
#
# # tt=np.prod(tuple(res))
# print(pres)
# print(pool)

# a=np.ones((3,4))
# # print(a)
# b=a.flatten()
# print(b)
# res=np.linalg.norm(b, ord=1)
# print(res)

# channels=3
# tensor = np.arange(60).reshape((2,3,5,2))
# print(tensor)
# tensor = tensor.reshape(-1, channels)
# print(tensor)

# x = np.ones((4, 2, 3))
# x=np.transpose(x, (0, 2, 1))
# print(x)
# a=np.arange(24).reshape((12,2))
# # print(a)
# b=a.reshape(3,4,2)
# sh=np.shape(b)
# print(a.shape)



class A:
    def __init__(self):
        pass
        # self.x=xx
        # self.y=yy
        # self.x=5
        # self.y=6
    def add(self):
        self.x=5
        self.y=6
        print("x和y的和为：%d"%(self.x+self.y))











