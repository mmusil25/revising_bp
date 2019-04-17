"""
Multi-Layer Neural Network iris classification

@author: Jian Meng

Project: Revised Backpropagation capstone

Date: 03/17/2019
"""

import numpy as np
import random
from numpy import exp, array, random, dot
from sklearn import datasets
from sklearn.model_selection import train_test_split


class NeuronLayer():
    def __init__(self, number_of_neurons, number_of_inputs_per_neuron):
        self.synaptic_weights = 2 * random.random((number_of_inputs_per_neuron, number_of_neurons)) - 1
        self.number_of_neurons = number_of_neurons
        self.number_of_inputs_per_neuron = number_of_inputs_per_neuron


class NeuralNetwork():
    def __init__(self, layer1, layer2, layer3):
        self.layer1 = layer1
        self.layer2 = layer2
        self.layer3 = layer3

    # The Sigmoid function, which describes an S shaped curve.
    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))

    # The derivative of the Sigmoid function.
    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    # The neural network thinks.
    def feed_forward(self, inputs):
        output_from_layer1 = self.__sigmoid(dot(inputs, self.layer1.synaptic_weights))
        output_from_layer2 = self.__sigmoid(dot(output_from_layer1, self.layer2.synaptic_weights))
        output_from_layer3 = self.__sigmoid(dot(output_from_layer2, self.layer3.synaptic_weights))
        print(f"output_from_layer3{output_from_layer3}")
        return output_from_layer1, output_from_layer2, output_from_layer3

    # The neural network prints its weights
    def print_weights(self):
        print("    Layer 1 (%d neurons, each with %d inputs): " % (self.layer1.number_of_neurons,
              self.layer1.number_of_inputs_per_neuron))
        print(self.layer1.synaptic_weights)

        print("    Layer 2 (%d neuron, with %d inputs): " % (self.layer2.number_of_neurons,
              self.layer2.number_of_inputs_per_neuron))
        print(self.layer2.synaptic_weights)


    # Neural Network Training phase
    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations, lr):
        for iteration in range(number_of_training_iterations):
            # Pass the training set through our neural network
            output_from_layer_1, output_from_layer_2, output_from_layer_3 = self.feed_forward(training_set_inputs)

            # Calculate the error for layer 3
            layer3_error = training_set_outputs - output_from_layer_3
            layer3_delta = layer3_error * self.__sigmoid_derivative(output_from_layer_3)

            # Calculate the error for layer 2
            layer2_error = layer3_delta.dot(self.layer3.synaptic_weights.T)
            layer2_delta = layer2_error * self.__sigmoid_derivative(output_from_layer_2)

            # Calculate the error of layer 1
            layer1_error = layer2_delta.dot(self.layer2.synaptic_weights.T)
            layer1_delta = layer1_error * self.__sigmoid_derivative(output_from_layer_1)

            # Increment of the update
            layer1_adjustment = training_set_inputs.T.dot(layer1_delta)
            layer2_adjustment = output_from_layer_1.T.dot(layer2_delta)
            layer3_adjustment = output_from_layer_2.T.dot(layer3_delta)

            # Adjust the weights
            self.layer1.synaptic_weights += lr * layer1_adjustment
            self.layer2.synaptic_weights += lr * layer2_adjustment
            self.layer3.synaptic_weights += lr * layer3_adjustment


def main():    
# Fetch the data from the iris data set: 
    iris = datasets.load_iris()
    X = iris.data[:, :4]
    Y = iris.target.reshape([150, 1])

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    print(f"ytrain.shape{y_train.shape}")
    #print(f"ytrain: {y_train}")
# # Create the first layer
#     layer1 = NeuronLayer(16, 3)
#
# # Create the second layer
#     layer2 = NeuronLayer(16, 16)
#
# # Create the third layer
#     layer3 = NeuronLayer(1, 16)

# Create the first layer
    layer1 = NeuronLayer(16, 4)

# Create the second layer
    layer2 = NeuronLayer(16, 16)

# Create the third layer
    layer3 = NeuronLayer(1, 16)

    neural_network = NeuralNetwork(layer1, layer2, layer3)
    print("Randomly initialize weights:")
    neural_network.print_weights()

# Training Phase
    neural_network.train(X_train, y_train, 1000, 0.8)
    print("New weights after training:")
    neural_network.print_weights()

# Test Phase
    count = 0
    for ii in range(X_test.shape[0]):
        _, _, output = neural_network.feed_forward(X_test[ii])
        print(f"A = {X_test[ii, 0]}, B = {X_test[ii, 1]}, C = {X_test[ii, 2]}")
        print(f"True output = {y_test[ii]}, Prediction = {output}\n")
        if output[0] - y_test[ii, 0] < 0.1:
            count += 1
    
    accuracy = count / np.shape(X_test)[0]
    print(f"The accuracy of the test set is:{accuracy * 100} %")


if __name__ == '__main__':
    main()
