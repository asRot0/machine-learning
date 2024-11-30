# Import required libraries
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load EMNIST dataset
dataset, info = tfds.load('emnist/balanced', as_supervised=True, with_info=True)
train_ds, test_ds = dataset['train'], dataset['test']

# Display dataset information
print(info)

"""
### Exploring the EMNIST Dataset
EMNIST (Extended MNIST) is a dataset of handwritten characters. The `balanced` split has 47 classes, 
including digits and uppercase/lowercase letters. This dataset is perfect for exploring both image classification 
and handling imbalanced class distributions.
"""

# View dataset structure
for example in train_ds.take(1):
    image, label = example
    print(f"Image shape: {image.shape}, Label: {label.numpy()}")

# EDA: Visualizing Label Distribution
label_counts = [label.numpy() for _, label in train_ds]
plt.figure(figsize=(14, 7))
sns.countplot(x=label_counts, palette="Set3", edgecolor="black")
plt.title("Distribution of Labels in EMNIST Balanced Dataset")
plt.xlabel("Class Labels")
plt.ylabel("Frequency")
plt.xticks(rotation=45)
plt.show()

"""
### Visualization Insight
The class distribution may not be uniform. Classes with fewer samples may affect model performance, making 
balancing techniques critical.
"""

# Visualizing some example images
def plot_emnist_samples(dataset, n_rows=3, n_cols=5):
    plt.figure(figsize=(15, 8))
    for i, (image, label) in enumerate(dataset.take(n_rows * n_cols)):
        plt.subplot(n_rows, n_cols, i + 1)
        plt.imshow(tf.squeeze(image), cmap="gray")
        plt.title(f"Label: {label.numpy()}")
        plt.axis("off")
    plt.tight_layout()
    plt.show()

plot_emnist_samples(train_ds, n_rows=4, n_cols=8)

"""
### Dataset Visualization
Above are some samples from the EMNIST dataset. Each image represents a character or digit. Notice the variance in 
handwriting styles.
"""

# Preprocessing the data
def preprocess(image, label):
    image = tf.expand_dims(image, axis=-1) / 255.0  # Normalize to [0, 1]
    label = tf.one_hot(label, depth=47)  # One-hot encode labels for 47 classes
    return image, label

train_ds = train_ds.map(preprocess).batch(32)
test_ds = test_ds.map(preprocess).batch(32)

# Inspect normalized pixel intensity distribution
sample_image = next(iter(train_ds))[0][0]  # First image from the batch
plt.figure(figsize=(10, 6))
sns.histplot(sample_image.numpy().ravel(), bins=50, kde=True, color="purple")
plt.title("Pixel Intensity Distribution in EMNIST (Normalized)")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")
plt.show()

"""
### Preprocessing Details
The images are normalized to bring pixel values into the range [0, 1], aiding model convergence. Labels are one-hot 
encoded to prepare for multi-class classification with 47 classes.
"""

# Build a CNN model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(47, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

"""
### CNN Model Architecture
1. **Conv2D + ReLU**: Extracts local patterns from images.
2. **MaxPooling2D**: Reduces spatial dimensions, decreasing computational cost.
3. **Dropout**: Regularization to prevent overfitting.
4. **Dense Layer**: Fully connected layers for high-level representation.
5. **Softmax Output**: Outputs probabilities across 47 classes.
"""

# Train the model
history = model.fit(train_ds, validation_data=test_ds, epochs=5)

# Plot training history
plt.figure(figsize=(12, 6))
plt.plot(history.history['accuracy'], label='Train Accuracy', marker='o')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='o')
plt.title("Model Accuracy Over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Train Loss', marker='o')
plt.plot(history.history['val_loss'], label='Validation Loss', marker='o')
plt.title("Model Loss Over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

"""
### Model Performance Analysis
Training and validation accuracy and loss curves show how well the model generalizes. If validation performance 
stalls or diverges, it may indicate overfitting.
"""

# Evaluate model
test_loss, test_accuracy = model.evaluate(test_ds)
print(f"Test Accuracy: {test_accuracy:.2f}")
