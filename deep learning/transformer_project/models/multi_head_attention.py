import tensorflow as tf
from tensorflow.keras.layers import Dense


class MultiHeadSelfAttention(tf.keras.layers.Layer):
    def __init__(self, embed_size, heads):
        super().__init__()
        assert embed_size % heads == 0, "Embedding size must be divisible by number of heads"

        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        # Linear layers for Q, K, V
        self.W_q = Dense(embed_size, use_bias=False)
        self.W_k = Dense(embed_size, use_bias=False)
        self.W_v = Dense(embed_size, use_bias=False)
        self.fc_out = Dense(embed_size)  # Final projection layer

    def call(self, values, keys, queries, mask=None):
        N = tf.shape(queries)[0]  # Batch size
        seq_length = tf.shape(queries)[1]

        # Linear projections
        queries = self.W_q(queries)
        keys = self.W_k(keys)
        values = self.W_v(values)

        # Reshape into multiple heads
        queries = tf.reshape(queries, (N, seq_length, self.heads, self.head_dim))
        keys = tf.reshape(keys, (N, seq_length, self.heads, self.head_dim))
        values = tf.reshape(values, (N, seq_length, self.heads, self.head_dim))

        # Transpose to (batch, heads, seq_len, head_dim)
        queries = tf.transpose(queries, perm=[0, 2, 1, 3])
        keys = tf.transpose(keys, perm=[0, 2, 1, 3])
        values = tf.transpose(values, perm=[0, 2, 1, 3])

        # Scaled Dot-Product Attention
        attention_scores = tf.matmul(queries, keys, transpose_b=True) / tf.math.sqrt(float(self.head_dim))

        # Apply mask to prevent attending to future tokens
        if mask is not None:
            attention_scores += (mask * -1e9)

        attention_weights = tf.nn.softmax(attention_scores, axis=-1)
        attention_output = tf.matmul(attention_weights, values)

        # Reshape and concatenate heads back
        attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3])
        attention_output = tf.reshape(attention_output, (N, seq_length, self.embed_size))

        return self.fc_out(attention_output)
