import numpy as np 
from func.activation_func import Sigmoid
from func.loss_func import SquareError

class MLP():

    def __init__(self, dims, activation_function=Sigmoid(), loss_func=SquareError()):
        self.weights = []
        self.baises = []
        for i in range(len(dims))[1:]:
            self.weights.append(np.random.rand(dims[i], dims[i-1]))
            self.baises.append(np.random.rand(dims[i], 1))

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
            self.baises[i] -= learning_rate * np.expand_dims(np.sum(self.delta[i], axis=1), -1)

    def sgd_with_layerwise_coeff(self, learning_rate=0.1, coeff=None):
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * coeff[i] * self.delta[i].dot(self.activation_outputs[i].T)
            self.baises[i] -= learning_rate * coeff[i] * np.expand_dims(np.sum(self.delta[i], axis=1), -1)

    def sgd_with_elementwise_coeff(self, learning_rate=0.1):
        for i in range(len(self.weights)):
            coeff = np.abs(self.delta[i]) / np.linalg.norm(self.delta[i])
            coeff = np.nan_to_num(coeff)
            coeff = np.sum(coeff, axis=1)
            self.weights[i] -= learning_rate * np.diag(coeff).dot(self.delta[i].dot(self.activation_outputs[i].T))
            self.baises[i] -= learning_rate * np.diag(coeff).dot(np.expand_dims(np.sum(self.delta[i], axis=1), -1))

    def train(self, X, y, learning_rate=0.1, optimizer='sgd', coeff=None):
        self.forward(X)
        err = self.loss_func.forward(self.activation_outputs[-1], y)
        loss_grad = self.loss_func.backward(self.activation_outputs[-1], y)
        self.backward(y, loss_grad)
        if optimizer == 'sgd':
            self.sgd(learning_rate=learning_rate)
        elif optimizer == 'sgd_with_layerwise_coeff':
            self.sgd_with_layerwise_coeff(learning_rate=learning_rate, coeff=coeff)
        elif optimizer == 'sgd_with_elementwise_coeff':
            self.sgd_with_elementwise_coeff(learning_rate=learning_rate)

        return err
    
    def predict(self, X):
        self.forward(X)
        return self.activation_outputs[-1]



