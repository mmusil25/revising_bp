"""
This network trains fully connected neural network
on the IRIS dataset using no special ML libraries.

This network is not my original work but has been borrowed as an educational resource.

Code Source:

https://github.com/rianrajagede/iris-python/blob/master/Python/iris_plain_mlp.py
"""
"""
SECTION 1 : Load and setup data for training
"""

import csv
import random
import math
import numpy as np
import matplotlib.pyplot as plt

random.seed(123)

# Load dataset
with open('./iris/iris.csv') as csvfile:
    csvreader = csv.reader(csvfile)
    next(csvreader, None)  # skip header
    dataset = list(csvreader)

# Change string value to numeric
for row in dataset:
    row[4] = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"].index(row[4])
    row[:4] = [float(row[j]) for j in range(len(row))]

# Split x and y (feature and target)
random.shuffle(dataset)
datatrain = dataset[:int(len(dataset) * 0.8)]
datatest = dataset[int(len(dataset) * 0.8):]
train_X = [data[:4] for data in datatrain]
train_y = [data[4] for data in datatrain]
test_X = [data[:4] for data in datatest]
test_y = [data[4] for data in datatest]

"""
SECTION 2 : Build and Train Model
Multilayer perceptron model, with one hidden layer.
input layer : 4 neuron, represents the feature of Iris
hidden layer : 3 neuron, activation using sigmoid
output layer : 3 neuron, represents the class of Iris
optimizer = gradient descent
loss function = Square ROot Error
learning rate = 0.005
epoch = 400
best result = 96.67%
"""


def matrix_mul_bias(A, B, bias):  # Matrix multiplication (for Testing)
    C = [[0 for i in range(len(B[0]))] for i in range(len(A))]
    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(B)):
                C[i][j] += A[i][k] * B[k][j]
            C[i][j] += bias[j]
    return C


def vec_mat_bias(A, B, bias):  # Vector (A) x matrix (B) multiplication
    C = [0 for i in range(len(B[0]))]
    for j in range(len(B[0])):
        for k in range(len(B)):
            C[j] += A[k] * B[k][j]
            C[j] += bias[j]
    return C


def mat_vec(A, B):  # Matrix (A) x vector (B) multipilicatoin (for backprop)
    C = [0 for i in range(len(A))]
    for i in range(len(A)):
        for j in range(len(B)):
            C[i] += A[i][j] * B[j]
    return C


def sigmoid(A, deriv=False):
    if deriv:  # derivation of sigmoid (for backprop)
        for i in range(len(A)):
            A[i] = A[i] * (1 - A[i])
    else:
        for i in range(len(A)):
            A[i] = 1 / (1 + math.exp(-A[i]))
    return A


# Define parameter
trial_num = "Test"
alfa = 0.005
epoch = 400
layer1_coeff = 3
layer2_coeff = layer1_coeff/2
neuron = [4, 5, 3]  # number of neuron each layer

# Initiate weight and bias with 0 value
weight = [[0 for j in range(neuron[1])] for i in range(neuron[0])]
weight_2 = [[0 for j in range(neuron[2])] for i in range(neuron[1])]
bias = [0 for i in range(neuron[1])]
bias_2 = [0 for i in range(neuron[2])]

# Initiate weight with random between -1.0 ... 1.0
for i in range(neuron[0]):
    for j in range(neuron[1]):
        weight[i][j] = 2 * random.random() - 1

for i in range(neuron[1]):
    for j in range(neuron[2]):
        weight_2[i][j] = 2 * random.random() - 1

cost_for_graph = []
for e in range(epoch):
    cost_total = 0
    for idx, x in enumerate(train_X):  # Update for each data; SGD

        # Forward propagation
        h_1 = vec_mat_bias(x, weight, bias)
        X_1 = sigmoid(h_1)
        h_2 = vec_mat_bias(X_1, weight_2, bias_2)
        X_2 = sigmoid(h_2)

        # Convert to One-hot target
        target = [0, 0, 0]
        target[int(train_y[idx])] = 1

        # Cost function, Square Root Eror
        eror = 0
        for i in range(3):
            eror += 0.5 * (target[i] - X_2[i]) ** 2
        cost_total += eror

        # Backward propagation
        # Update weight_2 and bias_2 (layer 2)
        delta_2 = []
        for j in range(neuron[2]):
            delta_2.append(-1 * (target[j] - X_2[j]) * X_2[j] * (1 - X_2[j]))

        for i in range(neuron[1]):
            for j in range(neuron[2]):
                weight_2[i][j] -= alfa * (delta_2[j] * X_1[i]) * layer2_coeff
                bias_2[j] -= alfa * delta_2[j] * layer2_coeff

        # Update weight and bias (layer 1)
        delta_1 = mat_vec(weight_2, delta_2)
        for j in range(neuron[1]):
            delta_1[j] = delta_1[j] * (X_1[j] * (1 - X_1[j]))

        for i in range(neuron[0]):
            for j in range(neuron[1]):
                weight[i][j] -= alfa * (delta_1[j] * x[i]) * layer1_coeff
                bias[j] -= alfa * delta_1[j] * layer1_coeff

    # store cost_total for graphing
    cost_total /= len(train_X)
    cost_for_graph.append(cost_total)
    if (e % 100 == 0):
        print("Epoch" , e/100, " out of ", epoch/100)
        print("Epoch cost: ", cost_total)

"""
SECTION 3 : Testing
"""

res = matrix_mul_bias(test_X, weight, bias)
res_2 = matrix_mul_bias(res, weight_2, bias)

# Get prediction
preds = []
for r in res_2:
    preds.append(max(enumerate(r), key=lambda x: x[1])[0])

# Print prediction
print("Predictions: ", preds)

# Calculate accuration
acc = 0.0
for i in range(len(preds)):
    if preds[i] == int(test_y[i]):
        acc += 1
netAcc = acc / len(preds) * 100
print("Network Accuracy: ", netAcc, "%")

"""
SECTION 4 : Plotting
"""

cost_for_graph = np.array(cost_for_graph)
#print(["cost_for_graph.shape: ", cost_for_graph.shape])
# Plot error over time
x_axis = np.arange(epoch)
#for i in range(epoch):
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(x_axis, cost_for_graph, 'r')
# axis(xmin, xmax, ymin, ymax)
plt.axis([0, epoch, 0, 0.40])
plt.xlabel('Training Epoch')
plt.ylabel('Error')
ax.set_title('MSE vs training iteration\n'
             ' Test Accuracy: %(val1)5.3f %% alpha: %(val2)5.3f\n'
             ' Layer 1 Coeff: %(val3)5.3f Layer 2 Coeff: %(val4)5.3f'
             % {'val1': netAcc, 'val2': alfa, 'val3': layer1_coeff, 'val4': layer2_coeff})
fname = "trial" + str(trial_num) + ".png" 
plt.savefig(fname)
