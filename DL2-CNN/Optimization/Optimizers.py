import numpy as np
class Sgd: #a way to update the weights
    def __init__(self,individual_delta):
        self.individual_delta=individual_delta
        return None
    def calculate_update(self, weight_tensor, gradient_tensor):
        weights=weight_tensor - self.individual_delta*gradient_tensor
        return weights #same as before
class SgdWithMomentum:
    def __init__(self,individual_delta,mu):
        self.v=None
        self.mu=mu
        self.eta=individual_delta
        return None
    def calculate_update(self, weight_tensor, gradient_tensor):
        if self.v is None:
            self.v=np.zeros_like(weight_tensor)
        self.v=self.mu*self.v-self.eta*gradient_tensor
        weights=weight_tensor+self.v
        return weights
class Adam:
    def __init__(self,individual_delta,mu,rho):
        self.v=None
        self.r=None
        self.mu=mu
        self.eta=individual_delta
        self.eps=0.0001
        self.rho=rho
        self.iteration=1 #first iteration
        return None
    def calculate_update(self, weight_tensor, gradient_tensor):
        if self.v is None:
            self.v=np.zeros_like(weight_tensor)
        if self.r is None:
            self.r=np.zeros_like(gradient_tensor)
        self.v=self.mu*self.v+(1-self.mu)*gradient_tensor
        self.r=self.rho*self.r+(1-self.rho)*np.multiply(gradient_tensor,gradient_tensor)
        vhat=self.v/(1-np.power(self.mu,self.iteration))
        rhat=self.r/(1-np.power(self.rho,self.iteration))
        weights=weight_tensor-self.eta*((vhat+self.eps)/(np.sqrt(rhat)+self.eps))
        self.iteration+=1 #updates the iteration number
        return weights
