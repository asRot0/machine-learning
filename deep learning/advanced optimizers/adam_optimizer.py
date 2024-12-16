'''
Adam Optimizer:
Combines the benefits of momentum and RMSProp for adaptive learning rate adjustments.

Formula:
  m = β1m + (1-β1)∇f(θ)
  v = β2v + (1-β2)(∇f(θ))²
  θ = θ - (η * m / √(v + ε))

Nadam: An enhancement over Adam with Nesterov momentum for faster convergence.

'''

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# Generate synthetic data
X = np.random.rand(1000, 10)
y = np.random.randint(0, 2, size=(1000,))

# Define a simple model
model = Sequential([
    Dense(32, activation='relu', input_shape=(10,)),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile model with Adam optimizer
adam_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=adam_optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X, y, epochs=20, batch_size=32, validation_split=0.2)
