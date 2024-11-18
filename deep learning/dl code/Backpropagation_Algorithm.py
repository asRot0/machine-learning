# Backpropagation Algorithm Explanation

import numpy as np


# Define activation function (sigmoid) and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)  # derivative of the sigmoid function


# Define input, expected output, and initial weights
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # input (XOR problem)
y = np.array([[0], [1], [1], [0]])  # expected output

# Set seed for reproducibility and initialize weights with small random numbers
np.random.seed(42)
weights_input_hidden = np.random.rand(2, 2) - 0.5
weights_hidden_output = np.random.rand(2, 1) - 0.5
learning_rate = 0.5
epochs = 10000

# Training with backpropagation
for epoch in range(epochs):
    # Forward pass: input to hidden layer
    hidden_input = np.dot(X, weights_input_hidden)  # weighted sum
    hidden_output = sigmoid(hidden_input)  # activation

    # Forward pass: hidden layer to output
    final_input = np.dot(hidden_output, weights_hidden_output)
    final_output = sigmoid(final_input)

    # Calculate error (Loss: Mean Squared Error)
    error = y - final_output
    if epoch % 1000 == 0:
        print(f'Epoch {epoch}, Error: {np.mean(np.square(error))}')

    # Backward pass: Output to hidden layer
    d_output = error * sigmoid_derivative(final_output)

    # Backward pass: Hidden layer to input layer
    error_hidden_layer = d_output.dot(weights_hidden_output.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_output)

    # Update weights (Gradient Descent)
    weights_hidden_output += hidden_output.T.dot(d_output) * learning_rate
    weights_input_hidden += X.T.dot(d_hidden_layer) * learning_rate

# Testing the network
print("\nFinal output after training:")
print(final_output)

# Explanation of Steps:
# 1. Forward pass: We compute the activations in the hidden and output layers.
# 2. Calculate loss/error: Here, MSE is used as the loss to measure the discrepancy.
# 3. Backward pass: Gradients for each layer are computed using the chain rule.
# 4. Weight updates: Gradient descent adjusts weights to minimize the error, repeating this in each epoch.
# This loop iteratively reduces the error and optimizes the neural network weights.

'''
This script covers a simple XOR neural network, showing how backpropagation iteratively updates weights to learn 
a pattern by minimizing error.
'''