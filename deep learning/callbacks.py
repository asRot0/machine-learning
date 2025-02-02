"""
Advanced TensorFlow/Keras Callbacks for Model Training
=====================================================
This script demonstrates various Keras callbacks, including:
- EarlyStopping: Stops training when performance degrades.
- ReduceLROnPlateau: Adjusts learning rate dynamically.
- ModelCheckpoint: Saves the best model during training.
- TensorBoard: Enables visualization of metrics.
- Custom Callbacks: Define custom training behavior.
"""

import tensorflow as tf
import numpy as np
import os

# Generate synthetic dataset
def create_dataset():
    X = np.random.rand(1000, 20)
    y = (np.sum(X, axis=1) > 10).astype(int)
    return X, y

X, y = create_dataset()
X_train, X_val = X[:800], X[800:]
y_train, y_val = y[:800], y[800:]

# Define a simple model
def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(20,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

model = create_model()

# =========================
# 1. Early Stopping
# =========================
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=5, restore_best_weights=True
)

# =========================
# 2. ReduceLROnPlateau
# =========================
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6
)

# =========================
# 3. Model Checkpoint
# =========================
checkpoint_path = "best_model.keras"
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, monitor='val_loss', save_best_only=True
)

# =========================
# 4. TensorBoard Logging
# =========================
log_dir = "logs/fit"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# =========================
# 5. Custom Callback
# =========================
class CustomLogger(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f"Epoch {epoch + 1}: Loss = {logs['loss']:.4f}, Val_Loss = {logs['val_loss']:.4f}")

custom_logger = CustomLogger()

# =========================
# Model Training with Callbacks
# =========================
model.fit(
    X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=32,
    callbacks=[early_stopping, reduce_lr, model_checkpoint, tensorboard_callback, custom_logger]
)
