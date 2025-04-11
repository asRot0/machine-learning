import tensorflow as tf
from tensorflow.keras.layers import Dense, Layer

class FeedForwardNetwork(Layer):
    def __init__(self, embed_size, expansion_factor=4, dropout_rate=0.1):
        super().__init__()
        self.ffn = tf.keras.Sequential([
            Dense(embed_size * expansion_factor, activation='relu'),
            Dense(embed_size),
            tf.keras.layers.Dropout(dropout_rate)
        ])

    def call(self, x, training=False):
        return self.ffn(x, training=training)
