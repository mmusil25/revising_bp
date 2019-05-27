import numpy as np 
from func.activation_func import Sigmoid
from func.loss_func import SquareError

class MLP():

    def __init__(self, dims, batch_size, activation_function=Sigmoid(), loss_func=SquareError()):
        self.weights = []
        self.baises = []
        for i in range(len(dims))[1:]:
            self.weights.append(np.random.rand(dims[i], dims[i-1]))
            self.baises.append(np.random.rand(dims[i], batch_size))

        self.activation_func = activation_function
        self.loss_func = loss_func

    def forward(self, X):
        self.pure_outputs = []
        self.activation_outputs = []
        self.activation_outputs.append(X)

        for i in range(len(self.weights)):
            self.pure_outputs.append(
                self.weights[i].dot(
                    self.activation_outputs[-1]) + self.baises[i])

            self.activation_outputs.append(
                self.activation_func.forward(
                    self.pure_outputs[-1]))


    def backward(self, y, loss_grad):
        self.delta = []
        self.delta.append(
            np.multiply(
                self.activation_func.backward(self.pure_outputs[-1]),
                loss_grad
            )
        )
        
        for i in range(len(self.weights) - 1)[::-1]:
            self.delta.append(
                np.multiply(
                    self.activation_func.backward(self.pure_outputs[i]),
                    self.weights[i+1].T.dot(self.delta[-1])
                )
            )

        self.delta = self.delta[::-1]

    def sgd(self, learning_rate=0.1):
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * self.delta[i].dot(self.activation_outputs[i].T)
            self.baises[i] -= learning_rate * self.delta[i]

    def train(self, X, y, learning_rate=0.1):
        self.forward(X)
        err = self.loss_func.forward(self.activation_outputs[-1], y)
        loss_grad = self.loss_func.backward(self.activation_outputs[-1], y)
        self.backward(y, loss_grad)
        self.sgd(learning_rate=0.1)
        return err
    
    def predict(self, X):
        self.forward(X)
        return self.activation_outputs[-1]



