from mlp import MLP
from sklearn.datasets import load_iris
import numpy as np 
import matplotlib.pyplot as plt

data = load_iris()
X = data['data'].T
y_ = data['target']

mlp = MLP(dims=[4, 8, 5, 3], batch_size=X.shape[1])

# Convert 'y_' to One-Hot label
y = np.zeros((len(y_), 3))
y[np.arange(len(y_)), y_] = 1
y = y.T

errs = []
for _ in range(500):
    errs.append(np.mean(mlp.train(X, y)))

plt.plot(list(range(500)), errs)
plt.show()