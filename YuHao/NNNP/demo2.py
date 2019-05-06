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

X_train, y_train = dataloader('train.dat')


dim, N = X_train.shape

mlp = MLP(dims=[dim, 5, 3, 2, 1], batch_size=N)

errs = []
for _ in range(500):
    err = mlp.train(X_train, y_train)
    errs.append(np.mean(err))

plt.plot(list(range(500)), errs)
plt.show()
