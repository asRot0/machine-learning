'''
RMSProp (Root Mean Square Propagation):
A variant of AdaGrad that uses an exponential moving average of squared gradients to address vanishing rates.

Formula:
  θ = θ - (η / √(E[∇f(θ)²])) ∇f(θ)

Advantages:
- Works well for recurrent neural networks and non-convex problems.
- Prevents the vanishing learning rate problem of AdaGrad.

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

# Compile model with RMSProp optimizer
rmsprop_optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001)
model.compile(optimizer=rmsprop_optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X, y, epochs=20, batch_size=32, validation_split=0.2)
