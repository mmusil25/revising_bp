"""Portland State University ECE Capstone 
[Group 14] 

Original author: Jian Meng, modified by Mark Musil

Description: A network which tests the error saturation prevention methodology. This example uses a simple XOR network.

"""
import numpy as np
import random
from numpy import exp, array, random, dot
import matplotlib.pyplot as plt



class NeuronLayer():
    def __init__(self, number_of_neurons, number_of_inputs_per_neuron):
        self.synaptic_weights = 2 * random.random((number_of_inputs_per_neuron, number_of_neurons)) - 1


class NeuralNetwork():
    def __init__(self, layer1, layer2):
        self.layer1 = layer1
        self.layer2 = layer2

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
        return output_from_layer1, output_from_layer2

    # The neural network prints its weights
    def print_weights(self):
        print("    Layer 1 (4 neurons, each with 2 inputs): ")
        print(self.layer1.synaptic_weights)
        print("    Layer 2 (1 neuron, with 4 inputs):")
        print(self.layer2.synaptic_weights)

    # Neural Network Training phase
    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations, lr):
        mse = np.zeros(number_of_training_iterations)  # Mean squared error
        for iteration in range(number_of_training_iterations):
            # Pass the training set through our neural network
            output_from_layer_1, output_from_layer_2 = self.feed_forward(training_set_inputs)

            # Calculate the error for layer 2
            layer2_error = training_set_outputs - output_from_layer_2
            mse[iteration] = np.mean(layer2_error**2)
            layer2_delta = layer2_error * self.__sigmoid_derivative(output_from_layer_2)

            # Calculate the error for layer 1
            layer1_error = layer2_delta.dot(self.layer2.synaptic_weights.T)
            layer1_delta = layer1_error * self.__sigmoid_derivative(output_from_layer_1)

            # Increment of the update
            layer1_adjustment = training_set_inputs.T.dot(layer1_delta)
            layer2_adjustment = output_from_layer_1.T.dot(layer2_delta)

            # Adjust the weights
            self.layer1.synaptic_weights += lr * layer1_adjustment
            self.layer2.synaptic_weights += lr * layer2_adjustment
        return np.mean(mse), mse


class Data_Generator():
    def __init__(self, data_size=None):
        if data_size == None:
            self.data_size = 100
        else:
            self.data_size = data_size

    def xor_generator(self):
        NN_input = np.zeros([self.data_size, 2])
        NN_target = np.zeros([self.data_size, 1])
        for ii in range(self.data_size):
            x1 = int(random.uniform(0, 2, 1))
            x2 = int(random.uniform(0, 2, 1))
            out = int(bool(x1 ^ x2))
            NN_input[ii, :] = np.array([x1, x2])
            NN_target[ii] = out
        return NN_input, NN_target


def main():
    random.seed(1)

    # Create the first layer (4 neurons, each with 2 inputs)
    layer1 = NeuronLayer(4, 2)

    # Create the second layer (a single neuron with 4 inputs)
    layer2 = NeuronLayer(1, 4)

    # Combine the layers to create a neural network
    neural_network = NeuralNetwork(layer1, layer2)
    # print(f"Randomly initialize weights: {neural_network.print_weights()}")
    print("Randomly initialize weights: \n")
    neural_network.print_weights()
    print("\n-------------------------------\n")

    # Generate the training set
    DATA_SIZE = 100
    training_data = Data_Generator(int(0.8 * DATA_SIZE))
    training_set_inputs, training_set_outputs = training_data.xor_generator()
    print(np.shape(training_set_inputs))
    # Training phase
    mse, mse_plot_array = neural_network.train(training_set_inputs, training_set_outputs, 1000, 0.1)
    print("Mean Squared Error after training: " + str(mse))
    #print(f"Stage 2) New weights after training: {neural_network.print_weights()}")
    print("Stage 2) New weights after training: \n")
    neural_network.print_weights()
    print("\n-------------------------------\n")

  #  Make a plot of the error versus training iterations
    x_data = np.arange(1000)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(x_data, mse_plot_array, 'r')
    ax.set_title('MSE vs training iteration\n Control')
    plt.show()

    # Test phase
    test_data = Data_Generator(int(0.2 * DATA_SIZE))
    test_set_inputs, test_set_outputs = test_data.xor_generator()
    for ii in range(np.shape(test_set_inputs)[0]):
        _, output = neural_network.feed_forward(test_set_inputs[ii])
        print(f"X1 = {test_set_inputs[ii, 0]}, X2 = {test_set_inputs[ii, 1]}")
        print(f"True output = {test_set_outputs[ii]}, Prediction = {output}")


if __name__ == '__main__':
    main()