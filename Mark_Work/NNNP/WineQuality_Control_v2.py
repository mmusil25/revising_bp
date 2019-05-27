from mlp import MLP
from sklearn.datasets import load_iris
import numpy as np 
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.decomposition import PCA 
import csv
import random

# Import dataset

#data = load_iris()
#X = data['data'].T
#y_ = data['target']

with open('./winequality-white.csv') as csvfile:
    csvreader = csv.reader(csvfile, delimiter = ";")
    next(csvreader, None)  # skip header
    dataset = list(csvreader)


# Change string value to numeric
for row in dataset:
    # row[4] = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"].index(row[4])
    row[:] = [float(row[j]) for j in range(len(row))]
    # print(["row", row])


# Split x and y (feature and target)
random.shuffle(dataset)
datatrain = dataset[:int(len(dataset) * 0.8)]
datatest = dataset[int(len(dataset) * 0.8):]
# train_X = [data[:11] for data in datatrain]
# train_y = [data[11] for data in datatrain]
test_X = [data[:11] for data in datatest]
test_y = [data[11] for data in datatest]


# Standardize the feature data

#train_X = np.array(train_X)
test_X = np.array(test_X)
scaler = preprocessing.StandardScaler()
#train_X = scaler.fit_transform(train_X)
test_X = scaler.fit_transform(test_X)
#train_X = train_X.tolist()
pca = PCA(n_components=8)
test_X = pca.fit_transform(test_X)
test_X = test_X.tolist()

# Hyperparamaters

networkDimensions = [4, 8, 5, 3]
learning_rate = 0.005
batch_size = 500

# Define the network

mlp = MLP(dims=networkDimensions, batch_size=batch_size)


# Train the network

errs = []
for _ in range(500):
    
    # Prepare the randomized training batch
    
    random.shuffle(datatrain)
    train_X = [data[:11] for data in datatrain[:batch_size]]
    y_ = [data[11] for data in datatrain[:batch_size]]
    train_X = np.array(train_X)
    train_X = scaler.fit_transform(train_X)
    train_X = pca.fit_transform(train_X)
    train_X = train_X.tolist()
    
    # Convert 'train_y' to One-Hot label
    train_y = np.zeros((len(y_), 3))
    train_y[np.arange(len(y_)), y_] = 1
    train_y = train_y.T
    
    # Do the training
    errs.append(np.mean(mlp.train(train_X, train_y, learning_rate = learning_rate)))
    
# Plot training error vs epoch

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(list(range(500)), errs)
plt.xlabel('Epoch')
plt.ylabel('Batch Error')
ax.set_title('Training results - Wine Quality'
             ' (Learning Rate: %(val1).4f, Batch size : %(val2)d)'
             % {'val1': learning_rate, 'val2': batch_size})
plt.show()


# Test the network

