import numpy as np
from layers import Layer
# Activation Functions

class ReLU(Layer):

    def forward_prop(Self, Z):

        self.Z = Z
        return np.maximum(0, Z)

    def backward_prop(self, dA, lr):

        return dA * (self.Z > 0)


#Sigmoid

class Sigmoid(Layer):

    def forward_prop(Self, Z, lr):

        self.A = (1 / ( 1 + np.exp(-Z))) 
        return self.A

    def backward_prop(self, dA, lr):

        return dA * (self.A * (1 - self.A))



#Softmax

class Softmax(Layer):

    def forward_prop(Self, Z):

        exp_Z = np.exp(Z - np.max(Z, axis=1, keepdim = True))
        self.A = exp_Z / np.sum(exp_Z, axis=1, keepdims=True)
        return self.A

    def backward_prop(self, dA, lr):

        return dA