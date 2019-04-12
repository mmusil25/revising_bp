from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from admm_nn import ADMM_Net
import torch
import numpy as np  


data = load_iris()
X = data['data']
y = data['target']


y_one_hot = np.zeros((len(y), 3))
y_one_hot[np.arange(len(y)), y] = 1

admm_net = ADMM_Net(dim_in=4, dim_hidden=[32, 16], dim_out=3, n_batch=150, beta=10.0, gamma=1.0)

admm_net.train(torch.from_numpy(X.T).float(), torch.from_numpy(y_one_hot.T).float())

y_hat_oh = admm_net.predict(torch.from_numpy(X.T).float())
y_hat = np.argmax(y_hat_oh, axis=0)
