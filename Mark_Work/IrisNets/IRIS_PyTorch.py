"""

This network trains fully connected neural network
on the IRIS dataset using PyTorch.


This code is based on a network written to train on MNIST. The strucuture of
the network is not my original work but the adaptions for the IRIS dataset are.
This code has been borrowed as an educational resource.

Code Source: https://www.kaggle.com/sdelecourt/cnn-with-pytorch-for-mnist
"""
import numpy as np # to handle matrix and data operation
import pandas as pd # to read csv and handle dataframe

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.autograd import Variable

from sklearn.model_selection import train_test_split
df = pd.read_csv('./iris/iris.csv')
# print(df.shape)
y = df['species'].values
X = df.drop(['species'],1).values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)
print(y_test.shape)

for i in range(len(y_train)):
    if y_train[i] == 'Iris-setosa':
        y_train[i] = 0
    elif y_train[i] == 'Iris-versicolor':
        y_train[i] = 1
    elif y_train[i] == 'Iris-virginica':
        y_train[i] = 2

for i in range(len(y_test)):
    if y_test[i] == 'Iris-setosa':
        y_test[i] = 0
    elif y_test[i] == 'Iris-versicolor':
        y_test[i] = 1
    elif y_test[i] == 'Iris-virginica':
        y_test[i] = 2

y_test = y_test.astype(float)
y_train = y_train.astype(float)
print(y_test)
print(y_train)
# Change string value to numeric
# for row in df:
#     row[4] = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"].index(row[4])
#     row[:4] = [float(row[j]) for j in range(len(row))]


BATCH_SIZE = 32
torch_X_train = torch.from_numpy(X_train).type(torch.LongTensor)
torch_y_train = torch.from_numpy(y_train).type(torch.LongTensor) # data type is long

# create feature and targets tensor for test set.
torch_X_test = torch.from_numpy(X_test).type(torch.LongTensor)
torch_y_test = torch.from_numpy(y_test).type(torch.LongTensor) # data type is long

# Pytorch train and test sets
train = torch.utils.data.TensorDataset(torch_X_train, torch_y_train)
test = torch.utils.data.TensorDataset(torch_X_test, torch_y_test)

# data loader
train_loader = torch.utils.data.DataLoader(train, batch_size = BATCH_SIZE, shuffle = False)
test_loader = torch.utils.data.DataLoader(test, batch_size = BATCH_SIZE, shuffle = False)


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(4, 5)
        self.linear2 = nn.Linear(5, 4)
        self.linear3 = nn.Linear(4, 3)

    def forward(self, X):
        X = torch.sigmoid(F.relu(self.linear1(X)))
        X = torch.sigmoid(F.relu(self.linear2(X)))
        X = self.linear3(X)
        return F.log_softmax(X, dim=1)


mlp = MLP()
print(mlp)


def fit(model, _train_loader):
    optimizer = torch.optim.Adam(model.parameters())#,lr=0.001, betas=(0.9,0.999))
    error = nn.CrossEntropyLoss()
    EPOCHS = 5
    model.train()
    for epoch in range(EPOCHS):
        correct = 0
        for batch_idx, (X_batch, y_batch) in enumerate(_train_loader):
            var_X_batch = Variable(X_batch).float()
            var_y_batch = Variable(y_batch)
            optimizer.zero_grad()
            output = model(var_X_batch)
            loss = error(output, var_y_batch)
            loss.backward()
            optimizer.step()

            # Total correct predictions
            predicted = torch.max(output.data, 1)[1]
            correct += (predicted == var_y_batch).sum()
            #print(correct)
            if batch_idx % 50 == 0:
                print('Epoch : {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t Accuracy:{:.3f}%'.format(
                    epoch, batch_idx*len(X_batch), len(train_loader.dataset), 100.*batch_idx / len(_train_loader),
                    torch.Tensor.item(loss.data), float(correct*100) / float(BATCH_SIZE*(batch_idx+1))))


fit(mlp, train_loader)


def evaluate(model):
    # model = mlp
    correct = 0
    for test_imgs, test_labels in test_loader:
        # print(test_imgs.shape)
        test_imgs = Variable(test_imgs).float()
        output = model(test_imgs)
        predicted = torch.max(output, 1)[1]
        predicted == test_labels
        correct += (predicted == test_labels).sum()
    print("Test accuracy:{:.3f}% ".format( float(correct) / (len(test_loader)*BATCH_SIZE)))


evaluate(mlp)

