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


    def train(self, X, y, n_epoch=500, n_warm_start=10):
        n_sample = X.shape[1]
        if n_sample % self.n_batch != 0:
            n_turn = int(n_sample / self.n_batch) + 1
        else:
            n_turn = n_sample // self.n_batch
        X_ = [X[:, i : (i + self.n_batch)] for i in self.n_batch * np.array(range(n_turn))]
        y_ = [y[:, i : (i + self.n_batch)] for i in self.n_batch * np.array(range(n_turn))]

        # Warm Start
        ws_acc = []
        for _ in range(n_warm_start):
            for i in range(len(X_)):
                self._a[0] = X_[i]
                self._update(y_[i], with_lambda=False)
                ws_acc.append(self._accuracy(y_[i], self.y_hat))
        print("Warm Start Finished. Accuracy: {}".format(sum(ws_acc)/len(ws_acc)))

        for ep in range(n_epoch):
            for i in range(len(X_)):
                self._a[0] = X_[i]
                self._update(y_[i])
                print("Epoch: {} - {}, Accuracy: {}".format(ep, i, self._accuracy(y_[i], self.y_hat)))


    def warm_start(self, X, y, n_epoch=10):
        n_sample = X.shape[1]
        if n_sample % self.n_batch != 0:
            n_turn = int(n_sample / self.n_batch) + 1
        else:
            n_turn = n_sample // self.n_batch
        X_ = [X[:, i : (i + self.n_batch)] for i in self.n_batch * np.array(range(n_turn))]
        y_ = [y[:, i : (i + self.n_batch)] for i in self.n_batch * np.array(range(n_turn))]

        # Warm Start
        ws_acc = []
        for _ in range(n_epoch):
            for i in range(len(X_)):
                self._a[0] = X_[i]
                self._update(y_[i], with_lambda=False)
                ws_acc.append(self._accuracy(y_[i], self.y_hat))
        print("Warm Start Finished. Accuracy: {}".format(sum(ws_acc)/len(ws_acc)))


    def predict(self, X):
        __a = []
        __a.append(X)
        __z = []
        for i in range(len(self.W)):
            __z.append(self.W[i].mm(__a[i]))
            __a.append(self.activation(__z[i]))

        return __z[-1]

    def _forward(self, X):
        self._a[0] = X
        for i in range(len(self.W)):
            self._z[i] = self.W[i].mm(self._a[i])
            self._a[i + 1] = self.activation(self._z[i])

        self.y_hat = self._z[-1]

        return self._z[-1]



    def _update(self, y, with_lambda=True):
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
                    ) + self.gamma[i] * self.activation(self._z[i])
                )
            )

            self._z_update_hidden_layer_relu(i)
        
        self.W[-1] = self._z[-1].mm(torch.pinverse(self._a[-2]))
        
        alpha = self.W[-1].mm(self._a[-2])

        self._z[-1] = (self.beta[-1] * alpha + y + self._lambda) / (1 + self.beta[-1])

        if with_lambda:
            self._lambda = self._lambda + self.beta[-1] * (self._z[-1] - alpha) 

        self.y_hat = self._z[-1]


    def _z_update_hidden_layer_relu(self, l):
        """
        There are two cases for ReLU function: 1. input >= 0; 2. otherwise
        """
        alpha = self.W[l].mm(self._a[l])

        opt_func = lambda z: (self.gamma[l] 
            * ((self._a[l + 1] 
            - (self.activation(z)) ** 2)) 
            + (self.beta[l] 
            * ((z - alpha) ** 2)))
            

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
        for i in range(_z.shape[0]):
            for j in range(_z.shape[1]):
                if min_decision[i, j]:
                    _z[i, j] = r1[i, j]
                else:
                    _z[i, j] = r2[i, j]

        self._z[l] = _z
  
        
    def _accuracy(self, y, y_hat):
        tf = (torch.argmax(y, dim=0) == torch.argmax(y_hat, dim=0))
        acc = torch.sum(tf).float() / tf.shape[0]

        return acc
        