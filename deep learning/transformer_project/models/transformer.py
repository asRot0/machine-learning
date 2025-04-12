import tensorflow as tf
from tensorflow.keras.layers import Embedding, Dropout, Dense
from .transformer_encoder import TransformerEncoderBlock
from .transformer_decoder import TransformerDecoderBlock
from .positional_encoding import PositionalEncoding

class Transformer(tf.keras.Model):
    def __init__(self,
                 num_layers,
                 embed_size,
                 num_heads,
                 ff_expansion,
                 input_vocab_size,
                 target_vocab_size,
                 max_seq_length,
                 dropout_rate=0.1):
        super().__init__()

        self.token_embedding_input = Embedding(input_vocab_size, embed_size)
        self.token_embedding_target = Embedding(target_vocab_size, embed_size)
        self.pos_encoding_input = PositionalEncoding(max_seq_length, embed_size)
        self.pos_encoding_target = PositionalEncoding(max_seq_length, embed_size)

        self.encoder_layers = [
            TransformerEncoderBlock(embed_size, num_heads, ff_expansion, dropout_rate)
            for _ in range(num_layers)
        ]

        self.decoder_layers = [
            TransformerDecoderBlock(embed_size, num_heads, ff_expansion, dropout_rate)
            for _ in range(num_layers)
        ]

        self.dropout_input = Dropout(dropout_rate)
        self.dropout_target = Dropout(dropout_rate)

        self.final_linear = Dense(target_vocab_size)

    def call(self, inputs, targets, enc_padding_mask=None, look_ahead_mask=None, dec_padding_mask=None, training=False):
        # Embed and add positional encoding to encoder inputs
        enc_embed = self.token_embedding_input(inputs)
        enc_embed += self.pos_encoding_input(enc_embed)
        enc_embed = self.dropout_input(enc_embed, training=training)

        # Pass through encoder stack
        for layer in self.encoder_layers:
            enc_embed = layer(enc_embed, mask=enc_padding_mask, training=training)

        # Embed and add positional encoding to decoder inputs
        dec_embed = self.token_embedding_target(targets)
        dec_embed += self.pos_encoding_target(dec_embed)
        dec_embed = self.dropout_target(dec_embed, training=training)

        # Pass through decoder stack
        for layer in self.decoder_layers:
            dec_embed = layer(dec_embed, enc_embed, look_ahead_mask, dec_padding_mask, training=training)

        # Final output layer: project decoder outputs to target vocab size
        final_output = self.final_linear(dec_embed)

        return final_output
