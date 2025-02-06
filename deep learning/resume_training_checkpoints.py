"""
Resume Training with Checkpoints in TensorFlow/Keras
=====================================================
This script demonstrates how to use Keras ModelCheckpoint to save model progress
and resume training after a break. This is useful when training deep learning models
on large datasets, where training can take several hours or days.

Key Features:
- Saves model weights after every epoch (or best weights only)
- Allows resuming training from the last saved state
- Uses TensorFlow/Keras ModelCheckpoint
"""

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np

# Simulated Dataset
num_samples, num_features = 1000, 20
num_classes = 10
train_data = np.random.rand(num_samples, num_features).astype(np.float32)
train_labels = np.random.randint(0, num_classes, size=(num_samples,))

# Define Model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation="relu", input_shape=(num_features,)),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(num_classes, activation="softmax")
])

# Compile Model
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Define Checkpoint Callback
checkpoint_path = "model_checkpoint.h5"
checkpoint_cb = ModelCheckpoint(
    filepath=checkpoint_path,
    save_best_only=True,  # Saves only the best model
    save_weights_only=True,  # Only saves weights, not full model
    verbose=1
)

# Train for Initial Period (e.g., First 5 Epochs)
print("\nStarting initial training...")
model.fit(train_data, train_labels, epochs=5, batch_size=32, callbacks=[checkpoint_cb])

# Simulate a break (Stopping the script)
print("\nTraining paused. You can stop the script here and resume later.")

# ---------------------------------------------
# Resuming Training (Later Session)
# ---------------------------------------------

# Load the saved weights before resuming
print("\nResuming training from last checkpoint...")
model.load_weights(checkpoint_path)

# Continue Training (Next 5 Epochs)
model.fit(train_data, train_labels, epochs=5, batch_size=32, callbacks=[checkpoint_cb])

print("\nTraining complete. Model saved at:", checkpoint_path)
