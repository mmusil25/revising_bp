from mlp import MLP
from sklearn.datasets import load_iris
import numpy as np 
import matplotlib.pyplot as plt

data = load_iris()
X = data['data'].T
y_ = data['target']

# Convert 'y_' to One-Hot label
y = np.zeros((len(y_), 3))
y[np.arange(len(y_)), y_] = 1
y = y.T

input_dim = X.shape[0]
N = X.shape[1]

mlp_lc = MLP(dims=[input_dim, 50, 20, 3])
mlp_base = MLP(dims=[input_dim, 50, 20, 3])


epochs = 500
errs_lc = []
errs_base = []
for _ in range(epochs):
    x_dim = X.shape[0]
    data = np.vstack((X,y))
    data = data.T
    np.random.shuffle(data)
    X = data.T[:x_dim, :]
    y = data.T[x_dim:, :]
    print(X.shape, y.shape)
    errs_lc.append(np.mean(mlp_lc.train(X, y, optimizer='sgd_with_elementwise_coeff', learning_rate=0.001)))
    errs_base.append(np.mean(mlp_base.train(X, y, optimizer='sgd', learning_rate=0.001)))

plt.plot(list(range(epochs)), errs_lc, label='SGD_LC')
plt.plot(list(range(epochs)), errs_base, label='SGD')
plt.legend()
plt.show()