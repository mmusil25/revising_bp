"""Portland State University ECE Capstone 
[Group 14] 

@author: Jian Meng
"""

import numpy as np

class Layer():
    def __init__(self, num_neurons, neuron_inputs):
        self.weights = 2 * np.random.random((neuron_inputs, num_neurons)) - 1 


class Neural_Nets():
    """Neural network structure
    Construct the two layers neural network and train it.
    """
    # Two layers network
    def __init__(self, layer1, layer2):
        self.layer1 = layer1
        self.layer2 = layer2
    
    # Acitvation function: Sigmoid
    def Sigmoid(self, x):
        activation = 1 / (1 + np.exp(-x))
        return activation

    # Derivative of Sigmoid
    def diff_Sigmoid(self, z):
        diff = z * (1 - z)
        return diff

    # Feed foward
    def forward(self, inputs):
        out_layer1 = self.Sigmoid(np.dot(inputs, self.layer1.weights))  # Input layer => Hiddenlayer1
        out_layer2 = self.Sigmoid(np.dot(inputs, self.layer1.weights))  # Hiddenlayer1 => Hiddenlayer2
        return out_layer1, out_layer2

    # Training: 
    def train(self, training_input, training_target, num_iter, lr):
        for iteration in range(num_iter):
            # Feed forward
            layer1_out, layer2_out = self.forward(training_input)

            # Hidden_Layer2 Error
            layer2_error = training_target - layer2_out
            layer2_diff = layer2_error * self.diff_Sigmoid(layer2_out)

            # Hidden_Layer1 Error
            layer1_error = np.dot(layer2_diff, self.layer2.weights.T)
            layer1_diff = layer1_error * self.diff_Sigmoid(layer1_out)

            # Adjustment
            layer1_adjustment = lr * training_input.T.dot(layer1_diff)
            layer2_adjustment = lr * layer1_out.T.dot(layer2_diff)

            # Update the weights
            self.layer1.weights += layer1_adjustment
            self.layer2.weights += layer2_adjustment

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

    # Create layer1 
    layer1 = Layer(4, 2)    # 4 neurons with 3 inputs

    # Create layer2
    layer2 = Layer(4, 4)    # 4 neurons with 4 inputs