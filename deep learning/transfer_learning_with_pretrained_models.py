"""
Transfer Learning with Pretrained Models
=========================================
This script demonstrates transfer learning using pretrained models from Keras, fine-tuning, and performance evaluation.
We will use the MobileNetV2 model pretrained on ImageNet as a feature extractor and fine-tune it for a custom classification task.
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# ======================================
# 1. Load and Preprocess the Dataset
# ======================================

# Load the CIFAR-10 dataset as an example
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Normalize pixel values
X_train = X_train / 255.0
X_test = X_test / 255.0

# Class names
class_names = [
    "Airplane", "Automobile", "Bird", "Cat", "Deer",
    "Dog", "Frog", "Horse", "Ship", "Truck"
]

print("\nDataset Information:")
print(f"Training Samples: {X_train.shape[0]}, Testing Samples: {X_test.shape[0]}")

# ======================================
# 2. Data Augmentation
# ======================================

data_augmentation = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

# Preview augmented images
plt.figure(figsize=(10, 10))
for X_batch, y_batch in data_augmentation.flow(X_train, y_train, batch_size=9, shuffle=False):
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(X_batch[i])
        plt.axis("off")
    break
plt.show()

# ======================================
# 3. Load Pretrained Model (Feature Extractor)
# ======================================

base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(32, 32, 3))

# Freeze base model layers
base_model.trainable = False

# Add custom classification head
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.5),
    layers.Dense(128, activation="relu"),
    layers.Dense(10, activation="softmax")  # CIFAR-10 has 10 classes
])

model.summary()

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# ======================================
# 4. Train the Model
# ======================================

history = model.fit(
    data_augmentation.flow(X_train, y_train, batch_size=32),
    validation_data=(X_test, y_test),
    epochs=10
)

# ======================================
# 5. Fine-Tune the Model
# ======================================

# Unfreeze some layers in the base model
for layer in base_model.layers[-50:]:
    layer.trainable = True

# Lower the learning rate for fine-tuning
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history_finetune = model.fit(
    data_augmentation.flow(X_train, y_train, batch_size=32),
    validation_data=(X_test, y_test),
    epochs=5
)

# ======================================
# 6. Evaluate the Model
# ======================================

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Accuracy: {test_acc:.4f}")

# Classification Report
y_pred = model.predict(X_test)
y_pred_classes = tf.argmax(y_pred, axis=1)
print("\nClassification Report:")
print(classification_report(y_test, y_pred_classes, target_names=class_names))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred_classes)
plt.figure(figsize=(10, 8))
plt.imshow(conf_matrix, cmap="Blues", interpolation="nearest")
plt.colorbar()
plt.xticks(range(10), class_names, rotation=45)
plt.yticks(range(10), class_names)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

# ======================================
# 7. Visualize Training Results
# ======================================

def plot_training_history(history, title="Training and Validation Accuracy"):
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(len(acc))

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, "b", label="Training Accuracy")
    plt.plot(epochs, val_acc, "r", label="Validation Accuracy")
    plt.title(f"{title} (Accuracy)")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, "b", label="Training Loss")
    plt.plot(epochs, val_loss, "r", label="Validation Loss")
    plt.title(f"{title} (Loss)")
    plt.legend()

    plt.show()

# Plot the results for initial training
plot_training_history(history, title="Training Results - Feature Extraction")

# Plot the results for fine-tuning
plot_training_history(history_finetune, title="Training Results - Fine-Tuning")


'''
Pretrained Models:
    - Uses MobileNetV2 pretrained on ImageNet.
    - Leverages transfer learning to adapt the model to a new task.
Feature Extraction:
    - Base model layers are frozen to act as a feature extractor.
Fine-Tuning:
    - Last 50 layers of the base model are unfrozen to fine-tune on the new dataset.
Data Augmentation:
    - Introduces variation in the dataset for better generalization.
Evaluation:
    - Generates a classification report and confusion matrix.
Visualization:
    - Plots training/validation accuracy and loss.
'''