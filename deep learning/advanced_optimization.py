"""
Advanced Optimization Techniques in Deep Learning
==================================================
This script demonstrates:
1. Gradient Clipping: Prevents exploding gradients in deep networks.
2. Learning Rate Schedules: Adaptive learning rates (Step Decay, Exponential Decay, Cosine Annealing, etc.).
3. Alternative Optimizers: RMSProp, SGD with Momentum, AdaGrad, and AdaBelief.
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


# =======================
# 1. Gradient Clipping
# =======================

def build_model_with_clipping():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(10,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)  # Clipping by norm
    model.compile(optimizer=optimizer, loss='mse')
    return model


# =============================
# 2. Learning Rate Schedules
# =============================

def lr_schedule(epoch):
    if epoch < 10:
        return 0.01
    elif epoch < 20:
        return 0.001
    else:
        return 0.0001


lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_schedule)

# Alternative: Exponential Decay
exponential_decay = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.01, decay_steps=1000, decay_rate=0.96, staircase=True
)

# Alternative: Cosine Annealing
cosine_decay = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=0.01, decay_steps=1000
)


# ============================
# 3. Alternative Optimizers
# ============================

def get_optimizers():
    return {
        "SGD with Momentum": tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
        "RMSProp": tf.keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9),
        "AdaGrad": tf.keras.optimizers.Adagrad(learning_rate=0.01),
        "AdaBelief": tf.keras.optimizers.Adam(learning_rate=0.001, amsgrad=True)
    }


# =============================
# 4. Visualization of Learning Rate Schedules
# =============================

def plot_lr_schedules():
    epochs = np.arange(30)
    lr_values = [lr_schedule(e) for e in epochs]
    exp_values = [exponential_decay(e) for e in epochs]
    cosine_values = [cosine_decay(e) for e in epochs]

    plt.figure(figsize=(8, 6))
    plt.plot(epochs, lr_values, label='Step Decay', marker='o')
    plt.plot(epochs, exp_values, label='Exponential Decay', marker='s')
    plt.plot(epochs, cosine_values, label='Cosine Annealing', marker='d')
    plt.xlabel("Epochs")
    plt.ylabel("Learning Rate")
    plt.title("Comparison of Learning Rate Schedules")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    model = build_model_with_clipping()
    optimizers = get_optimizers()
    plot_lr_schedules()
    print("Model with gradient clipping and alternative optimizers initialized.")
