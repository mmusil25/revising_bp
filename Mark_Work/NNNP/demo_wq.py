import numpy as np
import pandas as pd 
from mlp import MLP 
import matplotlib.pyplot as plt
import sys

df = pd.read_csv('./winequality-white.csv', sep=';')
data = df.values

# Allow bash to input parameters (dimensions, learning rate)
# Add filename as a parameters

X = data[:, :-1]
y = data[:, -1]
y = y.reshape((1, y.shape[0]))
X = X.T
input_dim = X.shape[0]
N = X.shape[1]

# Train-test split

batch_size = 50
div = batch_size * np.arange(int(N / batch_size))
data_list = np.array_split(X, div, axis=1)
data_list = data_list[1:-1]
y_list = np.array_split(y, div, axis=1)
y_list = y_list[1:-1]


mlp = MLP(dims=[input_dim, 10, 50, 10, 1], batch_size=batch_size)

errs = []
for _ in range(500):
    loc_errs = []
    for i in range(len(data_list)): 
        loc_errs.append(np.mean(mlp.train(data_list[i], y_list[i])))
    errs.append(np.mean(loc_errs))

# Test accuracy


np.save(loc_errs)

# Detailed plot
plt.plot(list(range(500)), errs)
plt.show()