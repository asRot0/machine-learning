import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize data
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Reshape for CNN
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

# One-hot encode labels
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

# Define CNN model
def get_model():
    model = keras.Sequential([
        keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# K-Fold Cross-Validation
k = 3
num_validation_samples = len(x_train) // k

# Shuffle data
indices = np.arange(len(x_train))
np.random.shuffle(indices)
x_train, y_train = x_train[indices], y_train[indices]

validation_scores = []

for fold in range(k):
    print(f"\nFold {fold+1}/{k}...")

    # Select validation and training data
    validation_data = x_train[num_validation_samples * fold : num_validation_samples * (fold + 1)]
    validation_labels = y_train[num_validation_samples * fold : num_validation_samples * (fold + 1)]

    training_data = np.concatenate((x_train[:num_validation_samples * fold], x_train[num_validation_samples * (fold + 1):]))
    training_labels = np.concatenate((y_train[:num_validation_samples * fold], y_train[num_validation_samples * (fold + 1):]))

    # Train model
    model = get_model()
    model.fit(training_data, training_labels, epochs=5, batch_size=64, verbose=1)

    # Evaluate model on validation set
    validation_score = model.evaluate(validation_data, validation_labels, verbose=0)[1]  # Get accuracy
    validation_scores.append(validation_score)

# Compute average validation score
final_validation_score = np.average(validation_scores)
print(f"\nFinal Cross-Validation Accuracy: {final_validation_score * 100:.2f}%")

# Train final model on full dataset
model = get_model()
model.fit(x_train, y_train, epochs=5, batch_size=64, verbose=1)

# Evaluate final model on test set
test_score = model.evaluate(x_test, y_test, verbose=1)[1]  # Get accuracy
print(f"\nFinal Test Accuracy: {test_score * 100:.2f}%")
