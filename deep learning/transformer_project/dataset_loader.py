import tensorflow as tf
import tensorflow_datasets as tfds

# Constants
MAX_SEQ_LEN = 40
BUFFER_SIZE = 20000
BATCH_SIZE = 64


def load_tokenizers():
    """Load or build tokenizers for German and English languages."""
    examples, _ = tfds.load("ted_hrlr_translate/de_to_en", split='train[:1%]', as_supervised=True)

    # Use SubwordTextEncoder for both source and target
    tokenizer_de = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
        (de.numpy() for de, _ in tfds.as_numpy(examples)), target_vocab_size=2 ** 13)

    tokenizer_en = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
        (en.numpy() for _, en in tfds.as_numpy(examples)), target_vocab_size=2 ** 13)

    return tokenizer_de, tokenizer_en


def encode_sentence(de, en, tokenizer_de, tokenizer_en):
    """Encode sentences into subword token sequences."""
    de_tokens = [tokenizer_de.vocab_size] + tokenizer_de.encode(de.numpy()) + [tokenizer_de.vocab_size + 1]
    en_tokens = [tokenizer_en.vocab_size] + tokenizer_en.encode(en.numpy()) + [tokenizer_en.vocab_size + 1]
    return de_tokens, en_tokens


def tf_encode(de, en, tokenizer_de, tokenizer_en):
    result_de, result_en = tf.py_function(
        func=lambda d, e: encode_sentence(d, e, tokenizer_de, tokenizer_en),
        inp=[de, en],
        Tout=[tf.int64, tf.int64]
    )
    result_de.set_shape([None])
    result_en.set_shape([None])
    return result_de, result_en


def filter_by_max_length(de, en, max_len=MAX_SEQ_LEN):
    return tf.logical_and(tf.size(de) <= max_len, tf.size(en) <= max_len)


def load_dataset():
    """Prepares the dataset and returns (train, val, tokenizer_de, tokenizer_en)."""
    tokenizer_de, tokenizer_en = load_tokenizers()

    examples, _ = tfds.load("ted_hrlr_translate/de_to_en", as_supervised=True, with_info=True)
    train_examples, val_examples = examples['train'], examples['validation']

    train_dataset = train_examples.map(lambda de, en: tf_encode(de, en, tokenizer_de, tokenizer_en))
    train_dataset = train_dataset.filter(filter_by_max_length)
    train_dataset = train_dataset.cache()
    train_dataset = train_dataset.shuffle(BUFFER_SIZE)
    train_dataset = train_dataset.padded_batch(BATCH_SIZE, padded_shapes=([None], [None]))
    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

    val_dataset = val_examples.map(lambda de, en: tf_encode(de, en, tokenizer_de, tokenizer_en))
    val_dataset = val_dataset.filter(filter_by_max_length)
    val_dataset = val_dataset.padded_batch(BATCH_SIZE, padded_shapes=([None], [None]))

    return train_dataset, val_dataset, tokenizer_de, tokenizer_en
