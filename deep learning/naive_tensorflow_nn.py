import tensorflow as tf
import numpy as np
import math
from tensorflow.keras.datasets import mnist
from tensorflow.keras import optimizers


# NaiveDense: A dense (fully connected) layer
class NaiveDense:
    def __init__(self, input_size, output_size, activation):
        self.activation = activation
        # Initialize weights and biases
        w_shape = (input_size, output_size)
        w_initial_value = tf.random.uniform(w_shape, minval=0, maxval=1e-1)
        self.W = tf.Variable(w_initial_value)

        b_shape = (output_size,)
        b_initial_value = tf.zeros(b_shape)
        self.b = tf.Variable(b_initial_value)

    def __call__(self, inputs):
        return self.activation(tf.matmul(inputs, self.W) + self.b)

    @property
    def weights(self):
        return [self.W, self.b]


# NaiveSequential: A simple model that chains multiple dense layers
class NaiveSequential:
    def __init__(self, layers):
        self.layers = layers

    def __call__(self, inputs):
        x = inputs
        for layer in self.layers:
            x = layer(x)
        return x

    @property
    def weights(self):
        weights = []
        for layer in self.layers:
            weights += layer.weights
        return weights


# BatchGenerator: A helper for generating batches of data
class BatchGenerator:
    def __init__(self, images, labels, batch_size=128):
        assert len(images) == len(labels)
        self.index = 0
        self.images = images
        self.labels = labels
        self.batch_size = batch_size
        self.num_batches = math.ceil(len(images) / batch_size)

    def next(self):
        images = self.images[self.index: self.index + self.batch_size]
        labels = self.labels[self.index: self.index + self.batch_size]
        self.index += self.batch_size
        return images, labels


# one_training_step: Computes loss and updates weights
def one_training_step(model, images_batch, labels_batch):
    with tf.GradientTape() as tape:
        predictions = model(images_batch)
        per_sample_losses = tf.keras.losses.sparse_categorical_crossentropy(labels_batch, predictions)
        average_loss = tf.reduce_mean(per_sample_losses)

    gradients = tape.gradient(average_loss, model.weights)
    update_weights(gradients, model.weights)
    return average_loss


learning_rate = 1e-3
optimizer = optimizers.SGD(learning_rate=learning_rate)


# update_weights: Uses gradient descent to adjust weights
def update_weights(gradients, weights):
    optimizer.apply_gradients(zip(gradients, weights))


# fit: Trains the model on the MNIST dataset
def fit(model, images, labels, epochs, batch_size=128):
    for epoch_counter in range(epochs):
        print(f"Epoch {epoch_counter + 1}")
        batch_generator = BatchGenerator(images, labels, batch_size)

        for batch_counter in range(batch_generator.num_batches):
            images_batch, labels_batch = batch_generator.next()
            loss = one_training_step(model, images_batch, labels_batch)

            if batch_counter % 100 == 0:
                print(f"Loss at batch {batch_counter}: {loss:.2f}")


def main():
    # Load MNIST dataset
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    train_images = train_images.reshape((60000, 28 * 28)).astype("float32") / 255
    test_images = test_images.reshape((10000, 28 * 28)).astype("float32") / 255

    # Define the model
    model = NaiveSequential([
        NaiveDense(input_size=28 * 28, output_size=512, activation=tf.nn.relu),
        NaiveDense(input_size=512, output_size=10, activation=tf.nn.softmax)
    ])

    # Train the model
    fit(model, train_images, train_labels, epochs=10, batch_size=128)

    # Evaluate the model
    predictions = model(test_images).numpy()
    predicted_labels = np.argmax(predictions, axis=1)
    accuracy = np.mean(predicted_labels == test_labels)
    print(f"Final accuracy: {accuracy:.2f}")


if __name__ == "__main__":
    main()
