import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Load a pretrained model (VGG16) and extract the first convolutional layer
model = tf.keras.applications.VGG16(weights="imagenet", include_top=False)
layer_name = "block1_conv1"  # First conv layer in VGG16
layer = model.get_layer(name=layer_name)

# Get the filters (weights) of the layer
filters, biases = layer.get_weights()

# Normalize filter values to be in 0-1 range for better visualization
filters_min, filters_max = filters.min(), filters.max()
filters = (filters - filters_min) / (filters_max - filters_min)

# Number of filters to visualize
num_filters = 6  # Change this to visualize more filters
fig, axes = plt.subplots(1, num_filters, figsize=(15, 5))

# Visualize each filter
for i in range(num_filters):
    ax = axes[i]
    f = filters[:, :, :, i]  # Extract i-th filter
    f = np.mean(f, axis=-1)  # Convert to grayscale
    ax.imshow(f, cmap="viridis")
    ax.axis("off")

plt.suptitle(f"Learned Filters from {layer_name}")
plt.show()
