import tensorflow as tf
from .models.transformer import Transformer
from .dataset_loader import load_dataset
import numpy as np

# Load dataset and tokenizers (we only need tokenizers here)
_, _, tokenizer_de, tokenizer_en = load_dataset()

# Rebuild the model (same hyperparams must match training!)
model = Transformer(
    num_layers=4,
    embed_size=128,
    num_heads=8,
    ff_expansion=4,
    input_vocab_size=tokenizer_de.vocab_size + 2,
    target_vocab_size=tokenizer_en.vocab_size + 2,
    max_seq_length=40,
    dropout_rate=0.1
)

# Load trained weights if available
model.load_weights('transformer_weights.h5')


def create_masks(input_seq, target_seq):
    # Padding mask for encoder
    enc_padding_mask = tf.cast(tf.math.equal(input_seq, 0), tf.float32)[:, tf.newaxis, tf.newaxis, :]

    # Look-ahead mask for decoder
    look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((target_seq.shape[1], target_seq.shape[1])), -1, 0)
    dec_target_padding_mask = tf.cast(tf.math.equal(target_seq, 0), tf.float32)[:, tf.newaxis, tf.newaxis, :]
    combined_mask = tf.maximum(look_ahead_mask, dec_target_padding_mask)

    # Padding mask for encoder output (used by decoder)
    dec_padding_mask = enc_padding_mask

    return enc_padding_mask, combined_mask, dec_padding_mask


def translate_sentence(sentence, max_length=40):
    # Tokenize German input
    sentence = [tokenizer_de.vocab_size] + tokenizer_de.encode(sentence) + [tokenizer_de.vocab_size + 1]
    encoder_input = tf.expand_dims(sentence, 0)

    # Start with <start> token for decoder
    output = tf.expand_dims([tokenizer_en.vocab_size], 0)  # <start> token

    for i in range(max_length):
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(encoder_input, output)

        # Make prediction
        predictions = model(
            encoder_input,
            output,
            enc_padding_mask=enc_padding_mask,
            look_ahead_mask=combined_mask,
            dec_padding_mask=dec_padding_mask,
            training=False
        )

        # Select the last token's prediction
        next_token = tf.argmax(predictions[:, -1:, :], axis=-1)

        # Append predicted token
        output = tf.concat([output, next_token], axis=-1)

        # Stop if <end> token is generated
        if next_token.numpy()[0][0] == tokenizer_en.vocab_size + 1:
            break

    # Remove <start> and <end> tokens, decode
    decoded_tokens = output.numpy()[0][1:-1]
    return tokenizer_en.decode(decoded_tokens)


# Example
if __name__ == "__main__":
    german_sentence = "Das ist ein gutes Beispiel."
    english_translation = translate_sentence(german_sentence)
    print("German:", german_sentence)
    print("Translated English:", english_translation)
