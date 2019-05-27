import numpy as np
import pandas as pd 
from mlp import MLP 
import matplotlib.pyplot as plt
 
df = pd.read_csv('./data/winequality-white.csv', sep=';')
data = df.values

X = data[:, :-1]
y = data[:, -1]
y = y.reshape((1, y.shape[0]))
X = X.T
input_dim = X.shape[0]
N = X.shape[1]

batch_size = 50
div = batch_size * np.arange(int(N / batch_size))
data_list = np.array_split(X, div, axis=1)
data_list = data_list[1:-1]
y_list = np.array_split(y, div, axis=1)
y_list = y_list[1:-1]


mlp_lc = MLP(dims=[input_dim, 20, 10, 1])
mlp_base = MLP(dims=[input_dim, 20, 10, 1])

epochs = 100
errs_lc = []
errs_base = []
for _ in range(epochs):
    loc_errs_lc = []
    loc_errs_base = []
    for i in range(len(data_list)): 
        loc_errs_lc.append(np.mean(mlp_lc.train(data_list[i], y_list[i], optimizer='sgd_with_elementwise_coeff', learning_rate=0.01)))
        loc_errs_base.append(np.mean(mlp_base.train(data_list[i], y_list[i], optimizer='sgd', learning_rate=0.01)))
    errs_lc.append(np.mean(loc_errs_lc))
    errs_base.append(np.mean(loc_errs_base))

plt.plot(list(range(epochs)), errs_lc, label='SGD_LC')
plt.plot(list(range(epochs)), errs_base, label='SGD')
plt.legend()
plt.show()