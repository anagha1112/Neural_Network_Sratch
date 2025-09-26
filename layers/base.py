# Base class (Blue print for Dense and Activation layers)

Class Layer:

    def forward_prop(Self, X):

        raise NotImplemetedError

    def backward_prop(Self, dA, lr = 0.01):

        raise NotImplemetedError

    def update_params(Self, lr):

        pass