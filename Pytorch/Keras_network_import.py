"""
Name: Mark Musil
Date: November 21, 2018

Project: Revised Backpropagation capstone

Description:

This is a simple CNN for MNIST digit recognition based of the design
at https://towardsdatascience.com/a-simple-2d-cnn-for-mnist-digit-recognition-a998dbc1e79a

This network was created to practice using Pytorch. It is based on a previous project
done in Keras, hence the name.

"""

from pathlib import Path
import requests

DATA_PATH = Path("data")
PATH = DATA_PATH/"mnist"
PATH.mkdir(parents=True, exist_ok=True)

URL = "http://deeplearning.net/data/mnist/"
FILENAME = "mnist.pkl.gz"

if not (PATH/FILENAME).exists():
    content = requests.get(URL + FILENAME).content
    (PATH/FILENAME).open("wb").write(content)

###############################################################################
# This dataset is in numpy array format, and has been stored using pickle,
# a python-specific format for serializing data.

import pickle
import gzip

with gzip.open((PATH / FILENAME).as_posix(), "rb") as f:
    ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")

###############################################################################
# Each image is 28 x 28, and is being stored as a flattened row of length
# 784 (=28x28). Let's take a look at one; we need to reshape it to 2d
# first.

from matplotlib import pyplot
import numpy as np
pyplot.imshow(x_train[0].reshape((28,28)), cmap = "gray")
print(x_train.shape)

###############################################################################
# PyTorch uses ``torch.tensor``, rather than numpy arrays, so we need to
# convert our data.

import torch
x_train, y_train, x_valid, y_valid = map(
    torch.tensor, (x_train, y_train, x_valid, y_valid)
)
n, c = x_train.shape
x_train, x_train.shape, y_train.min(), y_train.max()
print(x_train, y_train)
print(x_train.shape)
print(y_train.min(), y_train.m

###############################################################################
# Build the network class

class Mnist_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = torch.nn.Conv2d