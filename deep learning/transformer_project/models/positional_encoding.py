import tensorflow as tf
import numpy as np

class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, max_position, embed_size):
        super().__init__()
        self.pos_encoding = self.positional_encoding(max_position, embed_size)

    def get_angles(self, position, i, embed_size):
        # Calculate the angle rates for each position and dimension
        angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(embed_size))
        return position * angle_rates

    def positional_encoding(self, position, embed_size):
        angle_rads = self.get_angles(
            np.arange(position)[:, np.newaxis],
            np.arange(embed_size)[np.newaxis, :],
            embed_size
        )

        # Apply sin to even indices in the array; 2i
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

        # Apply cos to odd indices in the array; 2i+1
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        pos_encoding = angle_rads[np.newaxis, ...]  # Add batch dimension
        return tf.cast(pos_encoding, dtype=tf.float32)

    def call(self, x):
        # Add positional encoding to input embeddings
        return x + self.pos_encoding[:, :tf.shape(x)[1], :]
