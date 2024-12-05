import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic data
np.random.seed(42)
X = np.random.rand(1000, 20)  # 1000 samples, 20 features
y = np.random.randint(0, 2, 1000)  # Binary classification

# Activation functions to compare
activations = {
    "ReLU": tf.keras.layers.ReLU(),
    "Leaky ReLU": tf.keras.layers.LeakyReLU(alpha=0.1),
    "PReLU": tf.keras.layers.PReLU(shared_axes=[0, 1]),
    "ELU": tf.keras.layers.ELU(alpha=1.0),
    "SELU": tf.keras.layers.Activation("selu"),
    "RReLU (Randomized ReLU)": None  # Handled separately
}


# Custom Randomized ReLU
class RandomizedReLU(tf.keras.layers.Layer):
    def call(self, inputs, training=None):
        alpha_min, alpha_max = 0.1, 0.3
        alpha = tf.random.uniform(shape=tf.shape(inputs), minval=alpha_min, maxval=alpha_max)
        return tf.where(inputs > 0, inputs, alpha * inputs)


# Build a function to create models
def create_model(activation, input_dim=20, randomized_relu=False):
    model = Sequential()
    model.add(Dense(128, input_dim=input_dim))

    if randomized_relu:
        # Add Randomized ReLU layers manually
        model.add(RandomizedReLU())
        model.add(Dense(64))
        model.add(RandomizedReLU())
        model.add(Dense(32))
    else:
        # Add standard activation layers
        model.add(Dense(64, activation=activation))
        model.add(Dense(32, activation=activation))

    model.add(Dense(1, activation="sigmoid"))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


# Train models with each activation function
history_dict = {}
for name, activation in activations.items():
    print(f"Training with {name} activation...")
    model = create_model(activation, randomized_relu=(name == "RReLU (Randomized ReLU)"))
    history = model.fit(X, y, epochs=10, batch_size=32, verbose=0, validation_split=0.2)
    history_dict[name] = history

# Visualize results
plt.figure(figsize=(12, 6))
for name, history in history_dict.items():
    plt.plot(history.history['val_accuracy'], label=name)
plt.title("Validation Accuracy with Different Activation Functions")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
