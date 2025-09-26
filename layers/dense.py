import numpy as np
from layers import Layer

# Class for Dense Laayers

class Dense(Layer):

    def __init__(self, input_dim, output_dim):

        self.W = np.random.randn(input_dim, output_dim) * 0.01
        self.b = np.zeros((1, output_dim))

    def forward_prop(self, X):

        self.X = X
        return X @ self.W + self.b

    def backward_prop(Self, dZ):

        m = self.X.shape[0]
        self.dW = (self.X.T @ dZ) / m
        self.db = np.sum(dZ, axis = 0, keepdims = True) / m
        dA_prev = dZ @ self.W.T
        return dA_prev
        
    def update_params(self, lr):

        self.W -= lr * self.dW
        self.b -= lr * self.db




