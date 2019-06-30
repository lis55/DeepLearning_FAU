import numpy as np
from numpy import linalg as la

class L2_Regularizer:

    def __init__(self,alpha):
        self.alpha=alpha

    def calculate(self,weights):
        self.shrinkage_L2_d=self.alpha*weights   # shrinkage
        return self.shrinkage_L2_d

    def norm(self,weights):
        flatten_weights = weights.flatten()*self.alpha
        print('flatten_weights',flatten_weights.shape)
        self.norm_L2=np.linalg.norm(flatten_weights, ord=2, keepdims=True)
        return  self.norm_L2


class L1_Regularizer:

    def __init__(self,alpha):
        self.alpha=alpha

    def calculate(self, weights):
        sign_weights=np.sign(weights)
        self.shrinkage_L1_d=sign_weights*self.alpha    # shrinkage
        return self.shrinkage_L1_d

    def norm(self, weights):
        flatten_weights=weights.flatten()
        self.norm_L1=np.linalg.norm(flatten_weights, ord=1, keepdims=True)*self.alpha
        return self.norm_L1
