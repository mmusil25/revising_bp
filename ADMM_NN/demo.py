from sklearn.datasets import load_iris
from admm_nn import ADMM_Net
from torch_mlp import torch_mlp_demo
import torch
import numpy as np  


data = load_iris()
X = data['data']
y = data['target']

# torch_mlp_demo(X, y)

y_one_hot = np.zeros((len(y), 3))
y_one_hot[np.arange(len(y)), y] = 1

admm_net = ADMM_Net(dim_in=4, dim_hidden=[32, 16], dim_out=3, n_batch=150)

admm_net.train(torch.from_numpy(X).float(), torch.from_numpy(y_one_hot))
