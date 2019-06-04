import numpy as np 

class Sigmoid():
    def forward(self, X):
        return 1/(1 + np.exp(-X)) 
    
    def backward(self, X):
        return np.multiply(
            self.forward(X),
            (1 - self.forward(X))
        )

class ReLU():
    def forward(self, X):
        return X * (X > 0)

    def backward(self, X):
        return 1. * (X > 0)

