import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset and retrieve metadata
# tf_flowers dataset contains flower images and corresponding labels
dataset, info = tfds.load('tf_flowers', as_supervised=True, with_info=True)
dataset_size = info.splits['train'].num_examples  # Total number of examples in the dataset
class_names = info.features['label'].names  # List of class names
n_classes = info.features['label'].num_classes  # Number of unique classes

# Split the dataset into training, validation, and test sets
test_set_raw, valid_set_raw, train_set_raw = tfds.load(
    "tf_flowers",
    split=["train[:10%]", "train[10%:25%]", "train[25%:]"],  # 10% test, 15% validation, 75% training
    as_supervised=True)

# Clear previous session state to avoid conflicts
tf.keras.backend.clear_session()

# Define batch size for training and validation
batch_size = 32

# Preprocessing pipeline: Resize images and apply Xception's preprocessing function
preprocess = tf.keras.Sequential([
    tf.keras.layers.Resizing(height=224, width=224, crop_to_aspect_ratio=True),  # Resize images to 224x224
    tf.keras.layers.Lambda(tf.keras.applications.xception.preprocess_input)  # Apply Xception preprocessing
])

# Apply preprocessing to the datasets
train_set = train_set_raw.map(lambda X, y: (preprocess(X), y))
train_set = train_set.shuffle(1000, seed=42).batch(batch_size).prefetch(1)  # Shuffle and batch training set
valid_set = valid_set_raw.map(lambda X, y: (preprocess(X), y)).batch(batch_size)  # Preprocess validation set
test_set = test_set_raw.map(lambda X, y: (preprocess(X), y)).batch(batch_size)  # Preprocess test set

# Define data augmentation pipeline for additional variability in training data
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip(mode="horizontal", seed=42),  # Randomly flip images horizontally
    tf.keras.layers.RandomRotation(factor=0.05, seed=42),  # Slightly rotate images
    tf.keras.layers.RandomContrast(factor=0.2, seed=42)  # Adjust contrast of images
])

# Visualize augmented images
plt.figure(figsize=(12, 12))
for X_batch, y_batch in valid_set.take(1):
    X_batch_augmented = data_augmentation(X_batch, training=True)  # Apply augmentations
    for index in range(9):
        plt.subplot(3, 3, index + 1)
        # Rescale images for display and clip values to 0-1 range
        plt.imshow(np.clip((X_batch_augmented[index] + 1) / 2, 0, 1))
        plt.title(f"Class: {class_names[y_batch[index]]}")
        plt.axis("off")
plt.show()

# Set a random seed for reproducibility
tf.random.set_seed(42)

# Load the base Xception model pre-trained on ImageNet, excluding the top layer
base_model = tf.keras.applications.xception.Xception(weights="imagenet", include_top=False)

# Add a global average pooling layer and output layer for classification
avg = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
output = tf.keras.layers.Dense(n_classes, activation="softmax")(avg)
model = tf.keras.Model(inputs=base_model.input, outputs=output)

# Freeze all layers in the base model to retain pre-trained features
for layer in base_model.layers:
    layer.trainable = False

# Compile the model with an initial learning rate
optimizer = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9)
model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

# Train the model for a few epochs to fine-tune the top layers
history = model.fit(train_set, validation_data=valid_set, epochs=3)

# Print a summary of the base model's layer indices and names
for indices in zip(range(33), range(33, 66), range(66, 99), range(99, 132)):
    for idx in indices:
        print(f"{idx:3}: {base_model.layers[idx].name:22}", end="")
    print()

# Unfreeze a subset of layers in the base model for fine-tuning
for layer in base_model.layers[56:]:
    layer.trainable = True

# Re-compile the model with a lower learning rate for fine-tuning
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

# Train the model further to fine-tune the unfrozen layers
history = model.fit(train_set, validation_data=valid_set, epochs=10)
