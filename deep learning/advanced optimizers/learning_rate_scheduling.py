'''
Learning Rate Scheduling:
Dynamically adjusts the learning rate during training for better convergence.

Types:
- Step Decay: Reduce LR after specific intervals.
- Exponential Decay: Reduce LR exponentially.
- Cyclical LR: Oscillates LR between bounds.

'''

import tensorflow as tf
import numpy as np

# Example of Exponential Decay
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.01,
    decay_steps=1000,
    decay_rate=0.96
)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

# Apply optimizer to any model
