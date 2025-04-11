import tensorflow as tf
from tensorflow.keras.layers import LayerNormalization, Dropout
from .multi_head_attention import MultiHeadSelfAttention
from .feed_forward import FeedForwardNetwork

class TransformerDecoderBlock(tf.keras.layers.Layer):
    def __init__(self, embed_size, heads, ff_expansion=4, dropout_rate=0.1):
        super().__init__()
        self.self_attention = MultiHeadSelfAttention(embed_size, heads)
        self.enc_dec_attention = MultiHeadSelfAttention(embed_size, heads)

        self.norm1 = LayerNormalization(epsilon=1e-6)
        self.norm2 = LayerNormalization(epsilon=1e-6)
        self.norm3 = LayerNormalization(epsilon=1e-6)

        self.ffn = FeedForwardNetwork(embed_size, ff_expansion, dropout_rate)
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)
        self.dropout3 = Dropout(dropout_rate)

    def call(self, x, enc_output, look_ahead_mask=None, padding_mask=None, training=False):
        # Masked self-attention (for autoregressive generation)
        self_attn_output = self.self_attention(x, x, x, look_ahead_mask)
        self_attn_output = self.dropout1(self_attn_output, training=training)
        out1 = self.norm1(x + self_attn_output)

        # Encoder-Decoder attention
        enc_dec_attn_output = self.enc_dec_attention(enc_output, enc_output, out1, padding_mask)
        enc_dec_attn_output = self.dropout2(enc_dec_attn_output, training=training)
        out2 = self.norm2(out1 + enc_dec_attn_output)

        # Feed-forward network
        ffn_output = self.ffn(out2, training=training)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.norm3(out2 + ffn_output)

        return out3
