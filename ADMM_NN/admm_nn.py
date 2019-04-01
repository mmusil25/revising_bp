import torch
import numpy as np
from numpy.linalg import pinv, inv



relu = lambda x: torch.clamp(x, min=0.)

class ADMM_Net(object):
    def __init__(self, dim_in, dim_hidden, dim_out, n_batch=1, activation=relu, beta=1.0, gamma=10.0):
        """
        Note that W[0] means W_1 in paper, and a[0] means a_0. 
        """
        self.activation = activation
        self.n_layers = len(dim_hidden) + 1
        self.beta = torch.tensor([beta for _ in range(self.n_layers)])
        self.gamma = torch.tensor([gamma for _ in range(self.n_layers)])
        self.n_batch = n_batch

        self.W = []
        self.W.append(torch.randn(dim_hidden[0], dim_in))
        for i in range(len(dim_hidden))[1:]:
            self.W.append(torch.randn(dim_hidden[i], dim_hidden[i - 1]))
        self.W.append(torch.randn(dim_out, dim_hidden[-1]))

        self._lambda = torch.ones(dim_out, n_batch) 

        self._z = []
        self._a = []

        # Init. Activation Value and Pure Output
        self._a.append(torch.randn(dim_in, n_batch))

        for i in range(len(dim_hidden)):
            self._z.append(torch.randn(dim_hidden[i], n_batch))
            self._a.append(torch.randn(dim_hidden[i], n_batch))
        self._a.append(torch.randn(dim_out, n_batch))
        self._z.append(torch.randn(dim_out, n_batch))

        self.y_hat = None 


    def train(self, X, y, n_epoch=500, shuffle=False, n_warm_start=10):
        n_sample = X.shape[1]
        n_turn = int(n_sample / self.n_batch) + 1
        X_ = [X[:, i : (i + self.n_batch)] for i in self.n_batch * np.array(range(n_turn))]
        y_ = [y[:, i : (i + self.n_batch)] for i in self.n_batch * np.array(range(n_turn))]

        # Warm Start
        ws_acc = []
        for _ in range(n_warm_start):
            for i in range(len(X_)):
                self._update_without_lambda(y_[i])
                ws_acc.append(self._accuracy(y_[i], self.y_hat))
        print("Warm Start Finished. Accuracy: {}".format(sum(ws_acc)/len(ws_acc)))

        for ep in range(n_epoch):
            for i in range(len(X_)):
                self._update(y_[i])
                print("Epoch: {}, Batch:{}, Accuracy: {}".format(ep, i, self._accuracy(y_[i], self.y_hat)))
            

    def predict(self, X):
        return self._forward(X)


    def _forward(self, X):
        self._a[0] = X
        for i in range(len(self.W)):
            self._z[i] = self.W[i].mm(self._a[i])
            self._a[i + 1] = self.activation(self._z[i])

        self.y_hat = self._z[-1]

        return self.y_hat



    def _update_without_lambda(self, y):
        for i in range(self.n_layers - 1):
            self.W[i] = self._z[i].mm(torch.pinverse(self._a[i]))

            self._a[i + 1] = (
                torch.inverse(
                    self.beta[i + 1] 
                        * self.W[i + 1].t().mm(self.W[i + 1])
                        + self.gamma[i] * torch.eye(self.W[i + 1].size()[1]) 
                ).mm(
                    self.beta[i + 1] * self.W[i + 1].t().mm(
                        self._z[i + 1]
                    ) + self.gamma[i] * self._a[i + 1]
                )
            )

            self._z_update_hidden_layer_relu(i)
        
        self.W[-1] = self._z[-1].mm(torch.pinverse(self._a[self.n_layers - 2]))
        
        alpha = self.W[-1].mm(self._a[self.n_layers - 2])

        self._z[-1] = (self.beta[-1] * alpha + y - self._lambda) / (1 + self.beta[-1])

    def _update(self, y):
        self._update_without_lambda(y)
        self._lambda = self._lambda + self.beta[-1] * (self._z[-1] - alpha) 


    def _z_update_hidden_layer_relu(self, l):
        """
        There are two cases for ReLU function: 1. input >= 0; 2. otherwise
        """
        alpha = self.W[l].mm(self._a[l])

        opt_func = lambda z: torch.sum(
                self.gamma[l] * ((self._a[l + 1] - self.activation(z)) ** 2) 
                +
                self.beta[l] * ((z - alpha) ** 2) 
            )

        # Case 1: z >= 0:
        r1 = (self.gamma[l] + self.beta[l] * alpha) / (self.gamma[l] + self.beta[l])
        r1.clamp(min=0.)

        # Case 2: z < 0:
        r2 = alpha
        r2.clamp(max=0.)

        min_val1 = opt_func(r1)
        min_val2 = opt_func(r2)

        min_decision = min_val1 < min_val2

        _z = torch.zeros_like(r1)
        for i in range(len(_z)):
            if min_decision[i]:
                _z[i] = r1[i]
            else:
                _z[i] = r2[i]

        self._z[i] = _z
  
        
    def _accuracy(self, y, y_hat):
        tf = (torch.argmax(y, dim=0) == torch.argmax(y_hat, dim=0))
        acc = torch.sum(tf) / len(tf)

        return acc
        