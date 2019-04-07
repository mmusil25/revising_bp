
import math
import csv
import numpy as np
import sys
import random
sys.path.append('PBP_net/')
import PBP_net

np.random.seed(1)

# Load dataset
with open('/home/ieee-student/git/Me/revising_bp/Mark_Work/IrisNets/iris/iris.csv') as csvfile:
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
X_train = np.asarray([data[:4] for data in datatrain])
y_train = np.asarray([data[4] for data in datatrain])
X_test = np.asarray([data[:4] for data in datatest])
y_test = np.asarray([data[4] for data in datatest])

#********************* Old data import method
# We load the boston housing dataset
# data = np.loadtxt('../iris/iris.csv', delimiter=',')
# print(data.shape)
# We obtain the features and the targets
# X = data[ :, range(data.shape[ 1 ] - 1) ]
# y = data[ :, data.shape[ 1 ] - 1 ]
# We create the train and test sets with 90% and 10% of the data
# permutation = np.random.choice(range(X.shape[ 0 ]),
#     X.shape[ 0 ], replace = False)
# #size_train = np.round(X.shape[ 0 ] * 0.9)
# size_train = int(np.round(X.shape[ 0 ] * 0.9))
# index_train = permutation[ 0 : size_train ]
# index_test = permutation[ size_train : ]
#
# X_train = X[ index_train, : ]
# y_train = y[ index_train ]
# X_test = X[ index_test, : ]
# y_test = y[ index_test ]
#****************************


# We construct the network with one hidden layer with two-hidden layers
# with 50 neurons in each one and normalizing the training features to have
# zero mean and unit standard deviation in the trainig set.

n_hidden_units = 50
net = PBP_net.PBP_net(X_train, y_train,
    [ n_hidden_units, n_hidden_units ], normalize = True, n_epochs = 40)

# We make predictions for the test set

m, v, v_noise = net.predict(X_test)
print(["shape of m: ", m.shape])
# We compute the test RMSE

rmse = np.sqrt(np.mean((y_test - m)**2))

print (['RMSE:', rmse])

# We compute the test log-likelihood

test_ll = np.mean(-0.5 * np.log(2 * math.pi * (v + v_noise)) - \
    0.5 * (y_test - m)**2 / (v + v_noise))

print (['Test log-likelihood', test_ll])
