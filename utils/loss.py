import numpy as np

def cross_entropy_loss(y_true, y_pred):
    
    m = y_true.shape[0]
    eps = 1e-5

    y_pred = np.clip(y_pred, eps, 1-eps)

    loss = -np.sum(y_true * np.log(y_pred)) / m

    return loss
