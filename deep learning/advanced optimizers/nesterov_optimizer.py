'''
Nesterov Accelerated Gradient:
An improvement over Momentum optimization by computing gradients at a lookahead position.
Provides faster convergence for convex functions.

Formula:
  v = γv - η∇f(θ + γv)
  θ = θ + v

Advantages:
- Leads to better convergence compared to vanilla momentum.
- Performs a lookahead step before applying the gradient.

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

# Compile model with Nesterov Momentum optimizer
nesterov_optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(optimizer=nesterov_optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X, y, epochs=20, batch_size=32, validation_split=0.2)
