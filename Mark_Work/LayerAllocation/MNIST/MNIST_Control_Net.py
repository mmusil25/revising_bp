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
import mnist
import torchvision.datasets as datasets

mnist.init()

random.seed(123)

partition1 = 100
partition2 = 20

# Load dataset

train_X, train_y, test_X, test_y = mnist.load()
# print(["train_X, train_y, test_X, test_y", train_X.shape, train_y.shape, test_X.shape, test_y.shape])
train_X, train_y, test_X, test_y = train_X[0:partition1], train_y[0:partition1], test_X[0:partition2], test_y[0:partition2]
# print(["train_X, train_y, test_X, test_y", train_X.shape, train_y.shape, test_X.shape, test_y.shape])
# with open('./iris/iris.csv') as csvfile:
#     csvreader = csv.reader(csvfile)
#     next(csvreader, None)  # skip header
#     dataset = list(csvreader)

# mnist_trainset = datasets.MNIST(root='./MNIST_data', train=True, download=True, transform=None)
# mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=None)
# print(["mnist_trainset", mnist_trainset.data.numpy()])
# print(["mnist_testset", mnist_testset])
#

# # Change digit value to numeric
# print(["train_y.shape: ",train_y.shape])
train_y_onehot = np.zeros((train_y.shape[0], 10))
# print(["train_y_onehot.shape: ",train_y_onehot.shape])
for i in range(train_y.shape[0]):
     for j in range(10):
         if (j == train_y[i]):
            train_y_onehot[i][j] = 1
train_y = train_y_onehot

# print(["test_y.shape: ",test_y.shape])
test_y_onehot = np.zeros((test_y.shape[0], 10))
# print(["test_y_onehot.shape: ",test_y_onehot.shape])
for i in range(test_y.shape[0]):
     for j in range(10):
         if (j == test_y[i]):
            test_y_onehot[i][j] = 1
#test_y = test_y_onehot


# Split x and y (feature and target)
# random.shuffle(dataset)
# datatrain = dataset[:int(len(dataset) * 0.8)]
# datatest = dataset[int(len(dataset) * 0.8):]
# train_X = [data[:4] for data in datatrain]
# train_y = [data[4] for data in datatrain]
# test_X = [data[:4] for data in datatest]
# test_y = [data[4] for data in datatest]

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
    #print(["A:", A])
    #print(["len(A):", len(A)])
    if deriv:  # derivation of sigmoid (for backprop)
        for i in range(len(A)):
            A[i] = A[i] * (1 - A[i])
    else:
        for i in range(len(A)):
     #       print(["-A[i]: ",-A[i]])
            A[i] = 1 / (1 + math.exp(-np.max(A[i])))  #TODO: Fix the np.sum work around
    return A


# Define hyper parameters
trial_num = 1
alpha = 0.005
epoch = 50
neuron = [784, 30, 10]  # number of neuron each layer
write_out_name = "Trial" + str(trial_num) + ".txt"

f = open(write_out_name, "w+")
f.write(" alpha: %.4f, epoch: %d \n" % (alpha, epoch))
f.write(" neuron[0]: %d, neuron[1]: %d, neuron[2]: %d \n" %(neuron[0],
neuron[1], neuron[2]))
f.write("\n ##### Begin Training Input #####")

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
    # print(["Sub_Epoch: ", e])
    cost_total = 0
    for idx, x in enumerate(train_X):  # Update for each data; SGD
        # print(["x", x])
        # print(["idx", idx])
        # Forward propagation
        h_1 = vec_mat_bias(x, weight, bias)
        X_1 = sigmoid(np.clip(h_1, -500, 500))
        h_2 = vec_mat_bias(X_1, weight_2, bias_2)
        X_2 = sigmoid(h_2)
        #print(["X_2 dimensions", np.asarray(X_2).shape])
        # print(["X_2", X_2])
        # Convert to One-hot target
        # target = [0, 0, 0]
        # target[int(train_y[idx])] = 1

        # Cost function, Square Root Eror
        error = 0
        for i in range(3):
            error += 0.5 * (train_y[idx][i] - X_2[i]) ** 2
            # print(["target[i]", target[i]])
            # print(["train_y[idx][i]", train_y[idx][i]])
            # print(["X_2[i]", X_2[i]])
            # print(["error", error])
        cost_total += error

        # Backward propagation
        # Update weight_2 and bias_2 (layer 2)
        delta_2 = []
        for j in range(neuron[2]):
            delta_2.append(-1 * (train_y[j] - X_2[j]) * X_2[j] * (1 - X_2[j]))

        for i in range(neuron[1]):
            for j in range(neuron[2]):
                weight_2[i][j] -= alpha * (delta_2[j] * X_1[i])
                bias_2[j] -= alpha * delta_2[j]

        # Update weight and bias (layer 1)
        delta_1 = mat_vec(weight_2, delta_2)
        for j in range(neuron[1]):
            delta_1[j] = delta_1[j] * (X_1[j] * (1 - X_1[j]))

        for i in range(neuron[0]):
            for j in range(neuron[1]):
                weight[i][j] -= alpha * (delta_1[j] * x[i])
                bias[j] -= alpha * delta_1[j]

    # store cost_total for graphing
    cost_total /= partition1  # partition1 = len(train_X)
    # print(["cost_total", cost_total])
    cost_for_graph.append(cost_total)
    interval = 5
    if (e % interval == 0):
        print("Epoch" , e/interval, " out of ", epoch/interval)
        print("Epoch cost: ", cost_total)
        f.write("Epoch %.4f out of %.4f " % (e/interval, epoch/interval))
        f.write("Epoch cost: .4f " % cost_total)

print(["cost_for_graph", cost_for_graph])
cost_for_graph = np.array(cost_for_graph)
# print(["cost_for_graph.shape: ", cost_for_graph.shape])




"""
SECTION 3 : Testing
"""

# print(["test_X dimensions: ", np.asarray(test_X).shape])
# res = matrix_mul_bias(test_X, weight, bias)
# res_2 = matrix_mul_bias(res, weight_2, bias)
# print(["len(res_2)", len(res_2)])
# Get prediction

preds = []

for i in range(partition2):
    h_1 = vec_mat_bias(test_X[i], weight, bias)
    X_1 = sigmoid(np.clip(h_1, -500, 500))
    h_2 = vec_mat_bias(X_1, weight_2, bias_2)
    X_2 = sigmoid(h_2)
    max = np.argmax(X_2)
    #print(["X_2", X_2])
    preds.append(max)

# print(["preds", preds])

# for r in res_2:
#     print(["len(r)", len(r)])
#     print(["r", r])
#     print(["enumerate(r)", enumerate(r)])
#     print(["max(enumerate(r))", max(enumerate(r))])
#     preds.append(max(enumerate(r), key=lambda x: x[1])[0])

# Print prediction
print("Predictions: ", preds)
f.write("Predictions: \n")
f.write("[ ")
for i in range(len(preds)):
	f.write("%d, " % preds[i])
f.write(" ]")
# Calculate accuration
acc = 0.0
for i in range(len(preds)):
    if preds[i] == int(test_y[i]):
        acc += 1
    netAcc = acc / len(preds) * 100
print("Network Accuracy: ", netAcc, "%")
f.write("Network Accuracy: %.4f %%" % netAcc)

"""
SECTION 4 : Plotting
"""



# Plot error over time
x_axis = np.arange(epoch)
#for i in range(epoch):
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(x_axis, cost_for_graph, 'r')
# axis(xmin, xmax, ymin, ymax)
plt.axis([0, epoch, 0, 1.30])
plt.xlabel('Training iteration')
plt.ylabel('Error')
ax.set_title('MSE vs training iteration\n '
             'Layer Error Alloc. Control \n'
             'Network Accurary = %6.3f %% Alpha = %6.3f' % (netAcc, alpha))
#             ' (Error sat, alpha: %(val1)d, n: %(val2)d)'
#             % {'val1': alpha, 'val2': n})
# plt.show()
fname = "trial" + str(trial_num) + ".png" 
plt.savefig(fname)
