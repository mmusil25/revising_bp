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
from sklearn import preprocessing
from sklearn.decomposition import PCA 
import sys

random.seed(123)

# Load dataset
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


#partition1 = 2
#partition2 = 2
#train_X, train_y, test_X, test_y = train_X[0:partition1], train_y[0:partition1], test_X[0:partition2], test_y[0:partition2]

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
trial_num = sys.argv[1]   # "Test"
alfa = float(sys.argv[2]) # 0.005
epoch = int(sys.argv[3])  # 40
neuron = [8, int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]), 10]   # number of neuron each layer
batch_size = int(sys.argv[7])
write_out_name = str(trial_num) + ".txt"

f = open(write_out_name, "w+")
f.write(" Training sample count: %d, Test sample count: %d" % ( int(len(dataset) * 0.8)
, len(test_X)))
f.write(" alpha: %.4f, epoch: %d \n" % (alfa, epoch))
f.write(" neuron[0]: %d, neuron[1]: %d, neuron[2]: %d \n" % (neuron[0], neuron[1], neuron[2]))
f.write("###### Begin Training Output ###### \n")


# Initiate weight and bias with 0 value
weight = [[0 for j in range(neuron[1])] for i in range(neuron[0])]
weight_2 = [[0 for j in range(neuron[2])] for i in range(neuron[1])]
weight_3 = [[0 for j in range(neuron[3])] for i in range(neuron[2])]
weight_3 = [[0 for j in range(neuron[3])] for i in range(neuron[2])]
weight_4 = [[0 for j in range(neuron[4])] for i in range(neuron[3])]

bias = [0 for i in range(neuron[1])]
bias_2 = [0 for i in range(neuron[2])]
bias_3 = [0 for i in range(neuron[3])]
bias_4 = [0 for i in range(neuron[4])]

# Initiate weight with random between -1.0 ... 1.0
for i in range(neuron[0]):
    for j in range(neuron[1]):
        # weight[i][j] = random.gauss(0, 1)
        weight[i][j] = 2 * random.random() - 1
#print(["weight", weight]) 
#print(["weight dimension: ", [len(weight), len(weight[0])]])
      
for i in range(neuron[1]):
    for j in range(neuron[2]):
        weight_2[i][j] = 2 * random.random() - 1
        # weight_2[i][j] = random.gauss(0, 1)
#print(["weight_2", weight_2])
        
for i in range(neuron[2]):
    for j in range(neuron[3]):
        weight_3[i][j] = 2 * random.random() - 1
        #weight_3[i][j] = random.gauss(0, 1)
#print(["weight_3", weight_3])
for i in range(neuron[3]):
    for j in range(neuron[4]):
        weight_4[i][j] = 2 * random.random() - 1

cost_for_graph = []
for e in range(epoch):
    # print(["Epoch: ", epoch])
    random.shuffle(datatrain)
    #datatest = dataset[int(len(dataset) * 0.8):]
    train_X = [data[:11] for data in datatrain[:batch_size]]
    train_y = [data[11] for data in datatrain[:batch_size]]
    train_X = np.array(train_X)
    train_X = scaler.fit_transform(train_X)
    train_X = pca.fit_transform(train_X)
    train_X = train_X.tolist()
    # print(["train_X", train_X])
    cost_total = 0

    for idx, x in enumerate(train_X):  # Update for each data; SGD
        #print(["x", x])
        result = 0
        #for i in range(len(x)):
            #result += x[i]*weight[0][i]


        # print(["idx", idx])
        # print(["weight[1]: ", weight[1]])
        # print(["x", x])
        #print(["x*weight[1] ", result])
        # Forward propagation
        h_1 = vec_mat_bias(x, weight, bias)
        # print(["h_1", h_1])
        X_1 = sigmoid(np.clip(h_1, -500, 500))
        # print(["X_1", X_1])
        h_2 = vec_mat_bias(X_1, weight_2, bias_2)
        # print(["h_2", h_2])
        X_2 = sigmoid(h_2)
        # print(["X_2", X_2])
        h_3 = vec_mat_bias(X_2, weight_3, bias_3)
        # print(["h_3", h_3])
        X_3 = sigmoid(h_3)
        h_4 = vec_mat_bias(X_3, weight_4, bias_4)
        X_4 = sigmoid(h_4)
        # print(["X_3", X_3])
        # print(["y", train_y[idx]])
        
        # Convert to One-hot target
        target = [0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0]
        target[int(train_y[idx])] = 1

        # Cost function, Square Root Eror
        eror = 0
        for i in range(10):
            eror += 0.5 * (target[i] - X_4[i]) ** 2
        cost_total += eror

        # Backward propagation
        # Update weight_4 and bias_4 (layer 4)
        
        delta_4 = []
        for j in range(neuron[4]):
            delta_4.append(-1 * (target[j] - X_4[j]) * X_4[j] * (1 - X_4[j]))
        for i in range(neuron[3]):
            for j in range(neuron[4]):
                weight_4[i][j] -= alfa * (delta_4[j] * X_3[i])
                bias_4[j] -= alfa * delta_4[j]    
        #print(["weight_3", weight_3]) 
	# Update weight_3 and bias_3
        delta_3 = mat_vec(weight_4, delta_4)
        for j in range(neuron[3]):
            delta_3[j] = delta_3[j] * (X_3[j] * (1 - X_3[j]))

        for i in range(neuron[2]):
            for j in range(neuron[3]):
                weight_3[i][j] -= alfa * (delta_3[j] * X_2[i])
                bias_3[j] -= alfa * delta_3[j]

        # Update weight_2 and bias_2 (layer 2)
        delta_2 = mat_vec(weight_3, delta_3)
        for j in range(neuron[2]):
            delta_2[j] = delta_2[j] * (X_2[j] * (1 - X_2[j]))
                
        for i in range(neuron[1]):
            for j in range(neuron[2]):
                weight_2[i][j] -= alfa * (delta_2[j] * X_1[i])
                bias_2[j] -= alfa * delta_2[j]
        #print(["weight_2", weight_2]) 
        
        # Update weight and bias (layer 1)
        delta_1 = mat_vec(weight_2, delta_2)
        for j in range(neuron[1]):
            delta_1[j] = delta_1[j] * (X_1[j] * (1 - X_1[j]))

        for i in range(neuron[0]):
            for j in range(neuron[1]):
                weight[i][j] -= alfa * (delta_1[j] * x[i])
                bias[j] -= alfa * delta_1[j]
        # print(["weight", weight]) 
        # print(["weight dimension: ", [len(weight), len(weight[0])]])
        
    # store cost_total for graphing
    cost_total /= len(train_X)
    cost_for_graph.append(cost_total)
    interval = 10
    if (e % interval == 0):
        print("Epoch " , e/interval, " out of ", epoch/interval)
        print("Epoch cost: ", cost_total)
        f.write("Epoch " + str(e/interval) + " out of " + str(epoch/interval) + "\n")
        f.write("Epoch cost: %.5f \n" % cost_total)

cost_for_graph = np.array(cost_for_graph)
# print(["cost_for_graph.shape: ", cost_for_graph.shape])
# Plot error over time
x_axis = np.arange(epoch)
#for i in range(epoch):
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(x_axis, cost_for_graph, 'r')
# axis(xmin, xmax, ymin, ymax)
plt.axis([0, epoch, 0, 0.40])
plt.xlabel('Training iteration')
plt.ylabel('Error')
ax.set_title('MSE vs training iteration')
#             ' (Error sat, alpha: %(val1)d, n: %(val2)d)'
#             % {'val1': alpha, 'val2': n})
fname = "trial" + str(trial_num) + ".png" 
plt.savefig(fname)



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

# print("Predictions: ", preds)

# Calculate accuration
acc = 0.0
for i in range(len(preds)):
    if preds[i] == int(test_y[i]):
        acc += 1
print("Network Accuracy: ", acc / len(preds) * 100, "%")
f.write("Network Accuracy: %.4f" % (acc / len(preds) * 100) )
f.write("##### End of file #####")
