import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image

# Load the pretrained VGG16 model
model = tf.keras.applications.VGG16(weights="imagenet", include_top=False)

# Load and preprocess an image
img_path = tf.keras.utils.get_file('elephant.jpg',
                                   'https://upload.wikimedia.org/wikipedia/commons/6/69/June_odd-eyed-cat.jpg')
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
img_array = preprocess_input(img_array)  # Preprocess for VGG16

# Select first 3 convolutional layers
conv_layer_names = ["block1_conv1", "block2_conv1", "block3_conv1"]
layers = [model.get_layer(name).output for name in conv_layer_names]

# Create a model that outputs feature maps from these layers
activation_model = tf.keras.models.Model(inputs=model.input, outputs=layers)

# Get feature maps (activations)
feature_maps = activation_model.predict(img_array)

# Set number of filters to visualize per layer
num_filters = 6  # Change to visualize more filters

# Create a figure with 3 rows (for 3 layers) and 'num_filters' columns
fig, axes = plt.subplots(nrows=3, ncols=num_filters, figsize=(15, 10))

# Plot feature maps for each selected layer
for row, fmap in enumerate(feature_maps):  # Loop through 3 layers
    for col in range(num_filters):  # Loop through 'num_filters' feature maps
        ax = axes[row, col]
        ax.imshow(fmap[0, :, :, col], cmap="viridis")
        ax.axis("off")

# Add layer titles on the left side
layer_titles = ["Block 1 Conv 1", "Block 2 Conv 1", "Block 3 Conv 1"]
for ax, title in zip(axes[:, 0], layer_titles):
    ax.set_ylabel(title, rotation=90, size=14, labelpad=20)

plt.suptitle("Feature Maps from First 3 Conv Layers", size=16)
plt.tight_layout()
plt.show()
