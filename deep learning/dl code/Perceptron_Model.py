# Understanding Neural Networks and Perceptrons

# Import necessary libraries
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

# Generate a simple binary classification dataset
X, y = make_classification(n_samples=100, n_features=2, n_classes=2, random_state=42)

# Initialize and train the Perceptron model
perceptron = Perceptron()
perceptron.fit(X, y)

# Predict and evaluate
y_pred = perceptron.predict(X)
print("Perceptron Accuracy:", accuracy_score(y, y_pred))

# Detailed Explanation:
# This example demonstrates a single-layer perceptron, which is one of the simplest types of neural networks.
# The Perceptron model here is used to learn a linear decision boundary on the generated dataset.
# Accuracy is evaluated on the training data itself for demonstration purposes.
