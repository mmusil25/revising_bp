import numpy as np 

class SquareError():
    def forward(self, y_hat, y):
        return np.square(y_hat - y)
    
    def backward(self, y_hat, y):
        return 2 * (y_hat - y)

