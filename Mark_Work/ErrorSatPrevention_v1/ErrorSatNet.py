"""Portland State University ECE Capstone 
[Group 14] 

Original author: Jian Meng, modified by Mark Musil

Description: A network which tests the error saturation prevention methodology. This example uses a simple XOR network.

"""

import numpy as np


class Layer:

    def __init__(self, num_neurons, neuron_inputs):  # Outputs dim x input dim
        self.weights = 2 * np.random.random((neuron_inputs, num_neurons)) - 1


class XORNet:


    """Neural network structure
    Construct the three layer XOR neural network and train it.
    """
    # Two layers network
    def __init__(self):
        self.layer1 = Layer(3, 2)  # 3 outputs, 2 inputs
        self.layer2 = Layer(3, 3)
        self.layer3 = Layer(1, 3)

        # self.layer1 = 2 * np.random.random((3, 2)) - 1  # 3 outputs, 2 inputs
        # self.layer2 = 2 * np.random.random((3, 3)) - 1
        # self.layer3 = 2 * np.random.random((3, 3)) - 1
    
    # Activation function: sigmoid
    
    def sigmoid(self, x):
        return 1 + np.exp(-x)
    # Derivative of sigmoid
    # def diff_sigmoid(self, z):
    #    return activation

    # Derivative of sigmoid
    def diff_sigmoid(self, z): # If this network allows range of neurons to be [-1, 1] this may cause problems.
        return z * (1 - z)

    # Feed forward and return the hidden weights
    def forward_hidden(self, inputs):
        out_layer1 = self.sigmoid(np.dot(inputs, self.layer1.weights))  # Input layer => Hiddenlayer1
        out_layer2 = self.sigmoid(np.dot(out_layer1, self.layer2.weights))  # Hiddenlayer1 => Hiddenlayer2
        out_layer3 = self.sigmoid(np.dot(out_layer2, self.layer3.weights))
        return out_layer1, out_layer2, out_layer3

    def predict(self, input_array): # This should be used after the network is trained to make predictions on the data
        _, _, prediction = self.forward_hidden(input_array)
        return prediction

    # Training: 
    def train(self, training_input, training_target, num_iter, lr):
        for iteration in range(num_iter):

            # 1. Feed forward and get error
            layer1_out, layer2_out, layer3_out = self.forward_hidden(training_input[iteration])
            error = training_target[iteration] - layer3_out
            print('### Iter: ' + str(iteration) + ' ### Input: ' + str(training_input[iteration]) + ' Ground truth: ' +
                  str(training_target[iteration]) + ' Output: ' + str(layer3_out) + ' Error: ' + str(error))
            # 2. Backpropagation of error

            # Hidden layer 3 error
            layer3_diff = error * self.diff_sigmoid(layer3_out)

            # Hidden layer 2 error
            layer2_error = np.dot(layer3_diff, self.layer3.weights.T)
            layer2_diff = layer2_error * self.diff_sigmoid(layer2_out)

            # Hidden layer1 Error
            layer1_error = np.dot(layer2_diff, self.layer2.weights.T)
            layer1_diff = layer1_error * self.diff_sigmoid(layer1_out)

            # 3. Descend that gradient! (weight updates)

            # Adjustment
            layer1_adjustment = lr * training_input[iteration].T.dot(layer1_diff)
            layer2_adjustment = lr * layer1_out.T.dot(layer2_diff)
            layer3_adjustment = lr * layer2_out.T.dot(layer3_diff)
            # Update the weights
            self.layer1.weights += layer1_adjustment
            self.layer2.weights += layer2_adjustment
            self.layer3.weights += layer3_adjustment
# class Data_Generator():
#     def __init__(self, data_size, num_inputs):
#         self.data_size = data_size
#         self.num_inputs = num_inputs
    
#     def logic_xor(self, x1, x2):
#         if x1 > 1.5 or x2 > 1.5:
#             raise ValueError("The inputs must be less than 1.5")
#         elif x1 < 0.0 or x2 < 0.0:
#             raise ValueError("The inputs must be greater than 0")
        

#     def XOR_generator(self, noise_level):
#         """Generate the data sets and the target vector
#         """
#         Data_Set = np.zeros([self.num_inputs, self.data_size])
#         for ii in range(self.data_size):
#             pass

    
def main():

    # Randomness
    np.random.seed(1)

    # Make the training data

    training_examples = 1000
    x_train = np.zeros((training_examples, 2), dtype=int)
    y_train = np.ones(training_examples, dtype=int)
    for i in range(training_examples):
        # Generate two random numbers for XORing
        entry_one = np.random.randint(0, 1)
        entry_two = np.random.randint(0, 1)
        x_train[i, 0] = entry_one
        x_train[i, 1] = entry_two

        # Calculate xor of the two entries
        if (entry_one & entry_one) | (~entry_one & ~entry_two):
            y_train[i] = 0

    x_test = np.array([[0, 1],
                       [1, 1],
                       [0, 0],
                       [1, 0]])

    network = XORNet()
    num_iter = x_train.shape[0]
    learning_rate = 0.001
    network.train(x_train, y_train, num_iter, learning_rate)

    for i in range(x_test.shape[0]):
        print( 'Input: ' + str(x_test[i]) + ' Output: ' + str(network.predict(x_test[i])))


x = main()
