# Perceptron and Feedforward Network (basic_nn.py)

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Perceptron Model (Single-layer Neural Network)
class Perceptron:
    def __init__(self, input_dim, learning_rate=0.01, n_epochs=50):
        # Initialize weights randomly and set learning rate
        self.weights = np.random.randn(input_dim)
        self.bias = 0
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs

    def activation(self, x):
        # Activation function for binary classification
        return 1 if x >= 0 else 0

    def predict(self, X):
        # Generate predictions
        linear_output = np.dot(X, self.weights) + self.bias
        return np.array([self.activation(x) for x in linear_output])

    def train(self, X, y):
        # Train the model using perceptron learning rule
        for _ in range(self.n_epochs):
            for idx, x_i in enumerate(X):
                update = self.learning_rate * (y[idx] - self.predict([x_i])[0])
                self.weights += update * x_i
                self.bias += update

# Load dataset and train Perceptron
X, y = make_classification(n_samples=100, n_features=2, n_classes=2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
perceptron = Perceptron(input_dim=2)
perceptron.train(X_train, y_train)
predictions = perceptron.predict(X_test)
print("Perceptron Accuracy:", accuracy_score(y_test, predictions))


'''
Explanation:

This script demonstrates a single-layer neural network (Perceptron) with a step activation function, ideal for binary classification.
It trains using the perceptron rule and predicts outputs based on learned weights and bias.
'''