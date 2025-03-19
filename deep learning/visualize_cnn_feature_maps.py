import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models, datasets
from tensorflow.keras.models import Model

# Description:
# This script demonstrates the visualization of feature maps (filter activations)
# for the first convolutional layer in a simple CNN trained on the MNIST dataset.

# Load and preprocess MNIST data
(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
x_train = x_train[..., np.newaxis] / 255.0  # Normalize and add channel dimension
x_test = x_test[..., np.newaxis] / 255.0

# Define the input layer explicitly
inputs = layers.Input(shape=(28, 28, 1))

# Define a simple CNN architecture
x = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Conv2D(64, (3, 3), activation='relu')(x)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Flatten()(x)
x = layers.Dense(64, activation='relu')(x)
x = layers.Dense(10, activation='softmax')(x)

# Create the model
model = Model(inputs=inputs, outputs=x)

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=1, batch_size=64, validation_split=0.1)

# Extract the outputs of the Conv2D layers
layer_outputs = [layer.output for layer in model.layers if isinstance(layer, layers.Conv2D)]
activation_model = Model(inputs=model.input, outputs=layer_outputs)

# Pass a sample image through the model
sample_image = x_train[0:1]  # Shape: (1, 28, 28, 1)
activations = activation_model.predict(sample_image)

# Visualize the activations for the first Conv2D layer
first_layer_activation = activations[0]
num_filters = first_layer_activation.shape[-1]

# Plot feature maps for the first Conv2D layer
plt.figure(figsize=(15, 15))
for i in range(num_filters):
    ax = plt.subplot(6, 6, i + 1)  # Use 6x6 grid for visualizing filters
    plt.imshow(first_layer_activation[0, :, :, i], cmap="viridis")
    plt.axis("off")
plt.tight_layout()
plt.show()
