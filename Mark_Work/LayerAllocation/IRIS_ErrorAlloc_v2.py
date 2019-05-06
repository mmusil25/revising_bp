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
import sys

random.seed(123)

# Load dataset
with open('../iris/iris.csv') as csvfile:
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
layer_1_coeff = float(sys.argv[4]) #3
layer_2_coeff = layer_1_coeff/2
trial_num = sys.argv[1] #"Test"
alpha = float(sys.argv[2]) # 0.005
epoch = int(sys.argv[3]) # 40
neuron = [4, 5, 3]  # number of neuron each layer
write_out_name = "Trial" + str(trial_num) + ".txt"
f = open(write_out_name, "w+")
f.write(" Training sample count: %d, Test sample count: %d" % (len(train_X), len(test_X)))
f.write(" alpha: %.4f, epoch: %d \n" % (alpha, epoch))
f.write(" neuron[0]: %d, neuron[1]: %d, neuron[2]: %d \n" % (neuron[0], neuron[1], neuron[2]))
f.write(" Layer 1 Coefficient: %.4f, Layer 2 Coefficient: %.4f" % (layer_1_coeff, layer_2_coeff))
f.write("###### Begin Training Output ###### \n")


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
        # print(["target", target])

        # Cost function, Square Root Eror
        error = 0
        for i in range(neuron[2]):
            error += 0.5 * (target[i] - X_2[i]) ** 2
        #    print(["target[i]", target[i]])
        #    print(["train_y[i]", train_y[i]])
        #    print(["X_2[i]", X_2[i]])
        #    print(["error", error])
        cost_total += error




        # Backward propagation
        # Update weight_2 and bias_2 (layer 2)
        delta_2 = []
        for j in range(neuron[2]):
            delta_2.append(-1 * (target[j] - X_2[j]) * X_2[j] * (1 - X_2[j]))

        for i in range(neuron[1]):
            for j in range(neuron[2]):
                weight_2[i][j] -= alpha * (delta_2[j] * X_1[i]) * layer_2_coeff
                bias_2[j] -= alpha * delta_2[j] * layer_2_coeff

        # Update weight and bias (layer 1)
        delta_1 = mat_vec(weight_2, delta_2)
        for j in range(neuron[1]):
            delta_1[j] = delta_1[j] * (X_1[j] * (1 - X_1[j]))

        for i in range(neuron[0]):
            for j in range(neuron[1]):
                weight[i][j] -= alpha * (delta_1[j] * x[i]) * layer_1_coeff
                bias[j] -= alpha * delta_1[j] * layer_1_coeff

    # store cost_total for graphing
    cost_total /= len(train_X)
    print(["cost_total", cost_total])
    cost_for_graph.append(cost_total)
    interval = 10
    if (e % interval == 0):
        print("Epoch" , e/interval, " out of ", epoch/interval)
        print("Epoch cost: ", cost_total)
        f.write("Epoch " + str(e/interval) + " out of " + str(epoch/interval) + "\n")
        f.write("Epoch cost: %.5f \n" % cost_total)


#print(["cost_for_graph", cost_for_graph])
cost_for_graph = np.array(cost_for_graph)
#print(["cost_for_graph.shape: ", cost_for_graph.shape])



"""
SECTION 3 : Testing
"""
#print(["test_X dimensions: ", np.asarray(test_X).shape])
res = matrix_mul_bias(test_X, weight, bias)
#print(["len(res)", len(res)])
res_2 = matrix_mul_bias(res, weight_2, bias)
#print(["len(res_2)", len(res_2)])
# Get prediction
preds = []
for r in res_2:
    #print(["r", r])
    preds.append(max(enumerate(r), key=lambda x: x[1])[0])

# Print prediction
#print("Predictions: ", preds)

# Calculate accuration
acc = 0.0
for i in range(len(preds)):
    if preds[i] == int(test_y[i]):
        acc += 1
    netAcc = acc / len(preds) * 100
print("Network Accuracy: ", netAcc, "%")
f.write("Network Accuracy: %.4f" % (acc / len(preds) * 100) )
f.write("##### End of file #####")

"""
SECTION 4 : Plotting
"""


# cost_for_graph = np.array(cost_for_graph)
# print(["cost_for_graph.shape: ", cost_for_graph.shape])
# Plot error over time
x_axis = np.arange(epoch)
#for i in range(epoch):
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(x_axis, cost_for_graph, 'r')
# axis(xmin, xmax, ymin, ymax)
plt.axis([0, epoch, 0, 0.50])
plt.xlabel('Training iteration')
plt.ylabel('Error')
ax.set_title('MSE vs training iteration\n '
             'Layer Error Alloc. Control \n'
             'Network Accurary = %6.3f %% Alpha = %6.3f' % (netAcc, alpha))
#             ' (Error sat, alpha: %(val1)d, n: %(val2)d)'
#             % {'val1': alpha, 'val2': n})
fname = "trial" + str(trial_num) + ".png" 
plt.savefig(fname)
