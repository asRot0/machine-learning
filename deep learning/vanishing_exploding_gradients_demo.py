import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
import matplotlib.pyplot as plt

# Description:
'''
This script demonstrates the vanishing and exploding gradients problem 
using a simple Deep Neural Network (DNN). The network is trained with different
activation functions and initializations to observe the behavior of gradients.
'''

# Generate synthetic data
X = np.random.randn(1000, 20)
y = np.random.randint(0, 2, size=(1000, 1))

# Function to build a simple deep network
def build_dnn(input_dim, n_hidden=10, n_neurons=20, activation='relu', initializer='he_normal'):
    model = Sequential()
    model.add(Dense(n_neurons, activation=activation, kernel_initializer=initializer, input_shape=(input_dim,)))
    for _ in range(n_hidden - 1):
        model.add(Dense(n_neurons, activation=activation, kernel_initializer=initializer))
    model.add(Dense(1, activation='sigmoid'))  # Output layer for binary classification
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Build models with different activations and initializations
models = {
    "Sigmoid Init: Glorot": build_dnn(20, activation='sigmoid', initializer='glorot_uniform'),
    "Tanh Init: Glorot": build_dnn(20, activation='tanh', initializer='glorot_uniform'),
    "ReLU Init: He": build_dnn(20, activation='relu', initializer='he_normal'),
}

# Train and analyze gradients
history = {}
for name, model in models.items():
    print(f"\nTraining {name}...")
    h = model.fit(X, y, epochs=10, batch_size=32, verbose=0)
    history[name] = h.history

# Plot accuracy and loss curves
plt.figure(figsize=(12, 6))
for name, h in history.items():
    plt.plot(h['accuracy'], label=f"{name} Accuracy")
plt.title("Accuracy Across Activations and Initializations")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# Observing Gradients (Optional Advanced Analysis)
@tf.function
def get_gradients(model, inputs, targets):
    with tf.GradientTape() as tape:
        predictions = model(inputs, training=True)
        loss = model.compiled_loss(targets, predictions)
    return tape.gradient(loss, model.trainable_variables)

# Analyze gradients for each model
for name, model in models.items():
    gradients = get_gradients(model, tf.convert_to_tensor(X[:32], dtype=tf.float32),
                              tf.convert_to_tensor(y[:32], dtype=tf.float32))
    grad_norms = [np.linalg.norm(grad.numpy()) for grad in gradients if grad is not None]
    print(f"{name} - Gradient Norms: {grad_norms}")

'''
Detailed Description:
    - Vanishing Gradients: Shown using sigmoid and tanh activations with glorot_uniform initialization. 
      These activations squash input into small ranges, making gradients smaller.
    - Exploding Gradients: Highlighted with incorrect initialization or deeper networks where gradients 
      grow exponentially, leading to instability.
    - Solution Demonstrated:
        - ReLU with He Initialization: Stabilizes gradient propagation and avoids vanishing/exploding issues.
        - Adam Optimizer: Handles gradient issues by adaptive learning rates.
Notes:
    Replace print statements with proper logs for real-world projects.
    Enhance gradient analysis using gradient clipping or advanced visualization tools.
'''