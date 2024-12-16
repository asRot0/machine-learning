'''
AdaGrad (Adaptive Gradient):
Adjusts learning rate dynamically for each parameter based on the historical gradient.
Works well for sparse datasets but suffers from vanishing learning rates.

Formula:
  θ = θ - (η / √(Σ(∇f(θ))²)) ∇f(θ)

Advantages:
- Automatically adapts learning rates.
- Suitable for sparse data.

Disadvantage:
- Vanishing learning rates over time.

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

# Compile model with AdaGrad optimizer
adagrad_optimizer = tf.keras.optimizers.Adagrad(learning_rate=0.01)
model.compile(optimizer=adagrad_optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X, y, epochs=20, batch_size=32, validation_split=0.2)
