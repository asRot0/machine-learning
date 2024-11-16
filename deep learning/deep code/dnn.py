# Deep Neural Network (dnn.py)

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Create synthetic dataset
X, y = make_moons(n_samples=500, noise=0.1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Build a Deep Neural Network (DNN) model
model = Sequential([
    Dense(16, activation='relu', input_shape=(2,)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=10, verbose=1)
print("DNN Accuracy:", model.evaluate(X_test, y_test, verbose=0)[1])


'''
Explanation:

This model is a deep neural network with multiple hidden layers for binary classification.
Uses ReLU activations for hidden layers and sigmoid for the output, with binary_crossentropy as the loss function.
'''