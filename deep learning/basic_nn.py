# Basic Neural Network Implementation (basic_nn.py)

# Import necessary libraries
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


# Neural Network Architecture and Forward Propagation
class BasicNeuralNetwork:
    def __init__(self, input_dim, hidden_dim, output_dim):
        # Initialize weights and biases
        self.W1 = np.random.randn(input_dim, hidden_dim)
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, output_dim)
        self.b2 = np.zeros((1, output_dim))

    def sigmoid(self, z):
        # Sigmoid activation function
        return 1 / (1 + np.exp(-z))

    def forward(self, X):
        # Forward pass
        self.Z1 = X.dot(self.W1) + self.b1
        self.A1 = np.tanh(self.Z1)  # Using tanh as activation for hidden layer
        self.Z2 = self.A1.dot(self.W2) + self.b2
        output = self.sigmoid(self.Z2)  # Sigmoid for binary classification
        return output

    def predict(self, X):
        # Predict based on threshold
        probabilities = self.forward(X)
        return (probabilities > 0.5).astype(int)


# Create data and standardize
X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the network
nn = BasicNeuralNetwork(input_dim=2, hidden_dim=5, output_dim=1)
predictions = nn.predict(X_test)
print("Accuracy:", accuracy_score(y_test, predictions))


'''
Explanation of the Code:

The network initializes random weights, simulating a simplistic neural network setup.
Forward propagation is implemented using the tanh and sigmoid functions.
The class provides a predict method based on a threshold of 0.5, assuming binary classification.
'''