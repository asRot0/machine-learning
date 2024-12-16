'''
Momentum Optimization:
Accelerates gradient descent by using a velocity vector that accumulates past gradients with a momentum term.
It helps overcome local minima and reduces oscillations in the gradient direction.

Formula:
  v = γv - η∇f(θ)
  θ = θ + v

Where:
- v: velocity (momentum term)
- γ: momentum factor (commonly 0.9)
- η: learning rate

Advantages:
- Speeds up convergence in the relevant direction.
- Smoothens gradients, avoiding local oscillations.
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

# Compile model with Momentum optimizer
momentum_optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
model.compile(optimizer=momentum_optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X, y, epochs=20, batch_size=32, validation_split=0.2)
