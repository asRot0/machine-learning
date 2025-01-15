import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model

# Extract the first convolutional layer
layer_outputs = [layer.output for layer in model.layers if isinstance(layer, keras.layers.Conv2D)]
activation_model = Model(inputs=model.input, outputs=layer_outputs)

# Pass a sample image through the model
sample_image = x_train[0:1]  # Shape: (1, 28, 28, 1)
activations = activation_model.predict(sample_image)

# Visualize the filters' activations for the first layer
first_layer_activation = activations[0]
num_filters = first_layer_activation.shape[-1]

plt.figure(figsize=(15, 15))
for i in range(num_filters):
    ax = plt.subplot(6, 6, i + 1)
    plt.imshow(first_layer_activation[0, :, :, i], cmap="viridis")
    plt.axis("off")
plt.show()
