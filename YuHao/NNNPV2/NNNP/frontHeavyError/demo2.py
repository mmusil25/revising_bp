from mlp import MLP
import numpy as np 
import matplotlib.pyplot as plt

def dataloader(filepath):
    data = list()
    with open(filepath, 'r') as FILE:
        lines = FILE.readlines()
        for line in lines:
            float_str = line.lstrip().rstrip().split()
            data.append([float(_) for _ in float_str])

    data = np.array(data)
    X = data[:, 0 : 2]
    y = data[:, 2]
    return X.T, y

X, y = dataloader('./data/train.dat')
y = np.expand_dims(y, axis=0)
input_dim = X.shape[0]
N = X.shape[1]

mlp_lc = MLP(dims=[input_dim, 20, 10, 1])
mlp_base = MLP(dims=[input_dim, 20, 10, 1])


epochs = 100
errs_lc = []
errs_base = []
for _ in range(epochs):
    errs_lc.append(np.mean(mlp_lc.train(X, y, optimizer='sgd_with_elementwise_coeff', learning_rate=0.01)))
    errs_base.append(np.mean(mlp_base.train(X, y, optimizer='sgd', learning_rate=0.01)))

plt.plot(list(range(epochs)), errs_lc, label='SGD_LC')
plt.plot(list(range(epochs)), errs_base, label='SGD')
plt.legend()
plt.show()