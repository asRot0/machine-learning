"""
Complete Transformer Model Implementation
===========================================

This script implements a complete Transformer model using TensorFlow and Keras.
It includes the following components:
  - Multi-Head Self-Attention
  - Positional Encoding
  - Transformer Block (for the encoder)
  - Transformer Encoder
  - Transformer Classifier (Encoder-Only version)
  - Masked Multi-Head Self-Attention (for decoder)
  - Transformer Decoder Block
  - Transformer Decoder
  - Complete Transformer (Encoder + Decoder)

The script also contains dummy data generation and training examples.
"""

import tensorflow as tf
from tensorflow.keras.layers import Dense, LayerNormalization, Dropout, GlobalAveragePooling1D
import numpy as np


# --------------------------------------------
# Multi-Head Self-Attention Layer
# --------------------------------------------
class MultiHeadSelfAttention(tf.keras.layers.Layer):
    def __init__(self, embed_size, heads):
        super().__init__()
        assert embed_size % heads == 0, "Embedding size must be divisible by number of heads"

        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        # Linear layers for Q, K, V projections
        self.W_q = Dense(embed_size, use_bias=False)
        self.W_k = Dense(embed_size, use_bias=False)
        self.W_v = Dense(embed_size, use_bias=False)
        self.fc_out = Dense(embed_size)  # Final output projection

    def call(self, values, keys, queries, mask=None):
        N = tf.shape(queries)[0]  # Batch size
        seq_length = tf.shape(queries)[1]

        # Linear projections for Q, K, V
        queries = self.W_q(queries)
        keys = self.W_k(keys)
        values = self.W_v(values)

        # Split into multiple heads: shape becomes (batch, seq_length, heads, head_dim)
        queries = tf.reshape(queries, (N, seq_length, self.heads, self.head_dim))
        keys = tf.reshape(keys, (N, seq_length, self.heads, self.head_dim))
        values = tf.reshape(values, (N, seq_length, self.heads, self.head_dim))

        # Transpose for matmul: (batch, heads, seq_length, head_dim)
        queries = tf.transpose(queries, perm=[0, 2, 1, 3])
        keys = tf.transpose(keys, perm=[0, 2, 1, 3])
        values = tf.transpose(values, perm=[0, 2, 1, 3])

        # Scaled Dot-Product Attention
        attention_scores = tf.matmul(queries, keys, transpose_b=True) / tf.math.sqrt(float(self.head_dim))
        if mask is not None:
            attention_scores += (mask * -1e9)  # Apply mask (if provided)

        attention_weights = tf.nn.softmax(attention_scores, axis=-1)
        attention_output = tf.matmul(attention_weights, values)

        # Reassemble heads: transpose and reshape back to (batch, seq_length, embed_size)
        attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3])
        attention_output = tf.reshape(attention_output, (N, seq_length, self.embed_size))

        return self.fc_out(attention_output)


# --------------------------------------------
# Transformer Block for Encoder
# --------------------------------------------
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super().__init__()
        self.attention = MultiHeadSelfAttention(embed_size, heads)
        self.norm1 = LayerNormalization(epsilon=1e-6)
        self.norm2 = LayerNormalization(epsilon=1e-6)
        self.feed_forward = tf.keras.Sequential([
            Dense(forward_expansion * embed_size, activation="relu"),
            Dense(embed_size)
        ])
        self.dropout = Dropout(dropout)

    def call(self, value, key, query, mask=None):
        # Multi-Head Self-Attention with residual connection and layer norm
        attention = self.attention(value, key, query, mask)
        x = self.norm1(attention + query)
        # Feed-Forward Network with residual connection and layer norm
        forward = self.feed_forward(x)
        return self.norm2(forward + x)


# --------------------------------------------
# Positional Encoding Layer
# --------------------------------------------
class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, max_seq_length, embed_size):
        super().__init__()
        self.embed_size = embed_size

        # Create a matrix of shape (max_seq_length, embed_size) with positional encodings
        pos = np.arange(max_seq_length)[:, np.newaxis]
        div_term = np.exp(np.arange(0, embed_size, 2) * (-np.log(10000.0) / embed_size))
        pe = np.zeros((max_seq_length, embed_size))
        pe[:, 0::2] = np.sin(pos * div_term)
        pe[:, 1::2] = np.cos(pos * div_term)
        self.positional_encoding = tf.constant(pe[np.newaxis, :, :], dtype=tf.float32)

    def call(self, x):
        # Add positional encodings to input tensor
        return x + self.positional_encoding[:, :tf.shape(x)[1], :]


# --------------------------------------------
# Transformer Encoder
# --------------------------------------------
class TransformerEncoder(tf.keras.Model):
    def __init__(self, num_layers, embed_size, heads, forward_expansion, dropout, max_seq_length):
        super().__init__()
        self.embed_size = embed_size
        self.position_encoding = PositionalEncoding(max_seq_length, embed_size)
        self.layers = [TransformerBlock(embed_size, heads, dropout, forward_expansion) for _ in range(num_layers)]
        self.dropout = Dropout(dropout)

    def call(self, x, mask=None):
        x = self.position_encoding(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, x, x, mask)
        return x  # Final encoded representation


# --------------------------------------------
# Transformer Classifier (Encoder-only)
# --------------------------------------------
class TransformerClassifier(tf.keras.Model):
    def __init__(self, num_layers, embed_size, heads, forward_expansion, dropout, max_seq_length, num_classes):
        super().__init__()
        self.encoder = TransformerEncoder(num_layers, embed_size, heads, forward_expansion, dropout, max_seq_length)
        self.pooling = GlobalAveragePooling1D()
        self.fc_out = Dense(num_classes, activation="softmax")

    def call(self, x, mask=None):
        x = self.encoder(x, mask)
        x = self.pooling(x)
        return self.fc_out(x)


# --------------------------------------------
# Masked Multi-Head Self-Attention for Decoder
# --------------------------------------------
class MaskedMultiHeadSelfAttention(tf.keras.layers.Layer):
    def __init__(self, embed_size, heads):
        super().__init__()
        assert embed_size % heads == 0, "Embedding size must be divisible by number of heads"

        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        # Linear layers for Q, K, V projections
        self.W_q = Dense(embed_size, use_bias=False)
        self.W_k = Dense(embed_size, use_bias=False)
        self.W_v = Dense(embed_size, use_bias=False)
        self.fc_out = Dense(embed_size)  # Final output projection

    def call(self, values, keys, queries, mask=None):
        N = tf.shape(queries)[0]  # Batch size
        seq_length = tf.shape(queries)[1]

        queries = self.W_q(queries)
        keys = self.W_k(keys)
        values = self.W_v(values)

        queries = tf.reshape(queries, (N, seq_length, self.heads, self.head_dim))
        keys = tf.reshape(keys, (N, seq_length, self.heads, self.head_dim))
        values = tf.reshape(values, (N, seq_length, self.heads, self.head_dim))

        queries = tf.transpose(queries, perm=[0, 2, 1, 3])
        keys = tf.transpose(keys, perm=[0, 2, 1, 3])
        values = tf.transpose(values, perm=[0, 2, 1, 3])

        attention_scores = tf.matmul(queries, keys, transpose_b=True) / tf.math.sqrt(float(self.head_dim))
        if mask is not None:
            attention_scores += (mask * -1e9)  # Prevent attention to future tokens

        attention_weights = tf.nn.softmax(attention_scores, axis=-1)
        attention_output = tf.matmul(attention_weights, values)

        attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3])
        attention_output = tf.reshape(attention_output, (N, seq_length, self.embed_size))

        return self.fc_out(attention_output)


# --------------------------------------------
# Transformer Decoder Block
# --------------------------------------------
class TransformerDecoderBlock(tf.keras.layers.Layer):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super().__init__()
        self.masked_attention = MaskedMultiHeadSelfAttention(embed_size, heads)
        self.norm1 = LayerNormalization(epsilon=1e-6)
        self.attention = MultiHeadSelfAttention(embed_size, heads)  # Cross-attention
        self.norm2 = LayerNormalization(epsilon=1e-6)
        self.norm3 = LayerNormalization(epsilon=1e-6)
        self.feed_forward = tf.keras.Sequential([
            Dense(forward_expansion * embed_size, activation="relu"),
            Dense(embed_size)
        ])
        self.dropout = Dropout(dropout)

    def call(self, x, encoder_output, src_mask=None, tgt_mask=None):
        # Masked Self-Attention with residual connection
        masked_attention = self.masked_attention(x, x, x, tgt_mask)
        x = self.norm1(masked_attention + x)
        # Cross-Attention (encoder-decoder attention)
        attention = self.attention(encoder_output, encoder_output, x, src_mask)
        x = self.norm2(attention + x)
        # Feed Forward Network with residual connection
        forward = self.feed_forward(x)
        return self.norm3(forward + x)


# --------------------------------------------
# Transformer Decoder
# --------------------------------------------
class TransformerDecoder(tf.keras.Model):
    def __init__(self, num_layers, embed_size, heads, forward_expansion, dropout, max_seq_length):
        super().__init__()
        self.position_encoding = PositionalEncoding(max_seq_length, embed_size)
        self.layers = [TransformerDecoderBlock(embed_size, heads, dropout, forward_expansion) for _ in
                       range(num_layers)]
        self.dropout = Dropout(dropout)

    def call(self, x, encoder_output, src_mask=None, tgt_mask=None):
        x = self.position_encoding(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return x  # Final decoded representation


# --------------------------------------------
# Complete Transformer (Encoder + Decoder)
# --------------------------------------------
class Transformer(tf.keras.Model):
    def __init__(self, num_layers, embed_size, heads, forward_expansion, dropout, max_seq_length, num_classes):
        super().__init__()
        self.encoder = TransformerEncoder(num_layers, embed_size, heads, forward_expansion, dropout, max_seq_length)
        self.decoder = TransformerDecoder(num_layers, embed_size, heads, forward_expansion, dropout, max_seq_length)
        self.fc_out = Dense(num_classes, activation="softmax")

    def call(self, src, tgt, src_mask=None, tgt_mask=None):
        encoder_output = self.encoder(src, src_mask)
        decoder_output = self.decoder(tgt, encoder_output, src_mask, tgt_mask)
        # Use the last token's output for prediction
        return self.fc_out(decoder_output[:, -1, :])


# --------------------------------------------
# Main Training Section
# --------------------------------------------
if __name__ == "__main__":
    # Example 1: Transformer Classifier (Encoder-only)
    print("Training TransformerClassifier (Encoder-only)...")
    classifier_model = TransformerClassifier(
        num_layers=2,
        embed_size=128,
        heads=4,
        forward_expansion=4,
        dropout=0.1,
        max_seq_length=100,
        num_classes=10
    )
    classifier_model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    # Dummy dataset for classification: (batch_size, seq_length, embed_dim)
    X_train = np.random.rand(1000, 100, 128).astype(np.float32)
    y_train = np.random.randint(0, 10, size=(1000,))
    classifier_model.fit(X_train, y_train, epochs=5, batch_size=32)

    # Example 2: Complete Transformer (Encoder + Decoder)
    print("Training complete Transformer (Encoder + Decoder)...")
    transformer_model = Transformer(
        num_layers=2,
        embed_size=128,
        heads=4,
        forward_expansion=4,
        dropout=0.1,
        max_seq_length=100,
        num_classes=10
    )
    transformer_model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    # Dummy dataset for sequence-to-sequence tasks
    X_src = np.random.rand(1000, 100, 128).astype(np.float32)  # Source sequences
    X_tgt = np.random.rand(1000, 100, 128).astype(np.float32)  # Target sequences
    y_train = np.random.randint(0, 10, size=(1000,))
    transformer_model.fit([X_src, X_tgt], y_train, epochs=5, batch_size=32)
