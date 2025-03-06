import tensorflow as tf
from tensorflow import keras


class SimpleDense(keras.layers.Layer):
    def __init__(self, units, activation=None):
        super().__init__()
        self.units = units
        self.activation = activation

    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.W = self.add_weight(
            shape=(input_dim, self.units), initializer="random_normal"
        )
        self.b = self.add_weight(
            shape=(self.units,), initializer="zeros"
        )

    def call(self, inputs):
        y = tf.matmul(inputs, self.W) + self.b
        if self.activation is not None:
            y = self.activation(y)
        return y


# Instantiate the layer
my_dense = SimpleDense(units=32, activation=tf.nn.relu)

# Create some test inputs
input_tensor = tf.ones(shape=(2, 784))

# Call the layer on the inputs
output_tensor = my_dense(input_tensor)

# Print the output shape
print(output_tensor.shape)  # Expected output: (2, 32)
