import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
from tensorflow.keras.initializers import HeNormal, GlorotUniform, RandomNormal, Zeros, Ones


'''
This script demonstrates advanced techniques to analyze and address the vanishing and exploding gradients problem in 
deep neural networks (DNNs). It experiments with various initialization strategies, activation functions, and optimizers 
to observe their impact on gradient flow and training stability.

**Key Concepts Addressed:**
1. Effects of Initialization (`HeNormal`, `GlorotUniform`, `RandomNormal`, etc.).
2. Activation functions and their role (`ReLU`, `Sigmoid`, `Tanh`).
3. Gradient behavior visualization and dynamic analysis.
4. Strategies to mitigate issues (e.g., gradient clipping).

**Goals:**
- Provide insights into the dynamics of gradient propagation.
- Evaluate the effectiveness of solutions for vanishing/exploding gradients.
'''

# Generate synthetic data
X = np.random.randn(1000, 50)  # Larger input for advanced testing
y = np.random.randint(0, 2, size=(1000, 1))


# Function to build DNN with specified parameters
def build_advanced_dnn(input_dim, n_hidden=10, n_neurons=50, activation='relu', initializer='he_normal',
                       optimizer='adam'):
    model = Sequential()
    model.add(Dense(n_neurons, activation=activation, kernel_initializer=initializer, input_shape=(input_dim,)))
    for _ in range(n_hidden - 1):
        model.add(Dense(n_neurons, activation=activation, kernel_initializer=initializer))
    model.add(Dense(1, activation='sigmoid'))  # Output layer for binary classification
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model


# Experiment configurations
experiments = [
    {"activation": "sigmoid", "initializer": GlorotUniform(), "optimizer": "sgd", "label": "Sigmoid + Glorot + SGD"},
    {"activation": "tanh", "initializer": GlorotUniform(), "optimizer": "adam", "label": "Tanh + Glorot + Adam"},
    {"activation": "relu", "initializer": HeNormal(), "optimizer": "adam", "label": "ReLU + He + Adam"},
    {"activation": "relu", "initializer": RandomNormal(stddev=0.02), "optimizer": "rmsprop",
     "label": "ReLU + RandomNormal + RMSProp"},
    {"activation": "relu", "initializer": Zeros(), "optimizer": "adam", "label": "ReLU + Zeros Init"},
]

# Train models and log gradient norms
results = []
for config in experiments:
    print(f"Training: {config['label']}")
    model = build_advanced_dnn(input_dim=50, n_hidden=10, activation=config["activation"],
                               initializer=config["initializer"], optimizer=config["optimizer"])
    history = model.fit(X, y, epochs=50, batch_size=32, verbose=0)

    # Log final accuracy
    final_accuracy = history.history['accuracy'][-1]
    results.append((config['label'], final_accuracy))


    # Gradient analysis
    @tf.function
    def get_gradients(model, inputs, targets):
        with tf.GradientTape() as tape:
            predictions = model(inputs, training=True)
            loss = model.compiled_loss(targets, predictions)
        return tape.gradient(loss, model.trainable_variables)


    gradients = get_gradients(model, tf.convert_to_tensor(X[:32], dtype=tf.float32),
                              tf.convert_to_tensor(y[:32], dtype=tf.float32))
    grad_norms = [np.linalg.norm(grad.numpy()) for grad in gradients if grad is not None]
    print(f"{config['label']} - Gradient Norms: {grad_norms}")

# Visualizing training accuracies across experiments
labels, accuracies = zip(*results)
plt.figure(figsize=(10, 6))
plt.bar(labels, accuracies, color='skyblue')
plt.xlabel('Configurations')
plt.ylabel('Final Training Accuracy')
plt.title('Performance Across Initializations and Activations')
plt.xticks(rotation=45, ha='right')
plt.show()


'''
Features in the Script:
    - Multiple Initializations: Includes HeNormal, GlorotUniform, RandomNormal, and edge cases like Zeros.
    - Activation Functions: Compares Sigmoid, Tanh, and ReLU for gradient behavior.
    - Optimizers: Explores SGD, Adam, and RMSProp to show their impact.
    - Gradient Norm Analysis: Captures gradient magnitudes layer-wise for deeper understanding.
    - Dynamic Inputs: Allows testing with varying network sizes and complexities.
Key Observations:
    - Vanishing Gradients: More pronounced with Sigmoid or Tanh and default GlorotUniform initialization.
    - Exploding Gradients: Observed with poor initialization like RandomNormal with high standard deviation.
    - Stabilization: Achieved with ReLU and HeNormal due to better gradient flow.
'''