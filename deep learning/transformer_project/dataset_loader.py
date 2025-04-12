import tensorflow as tf
import tensorflow_datasets as tfds

# Constants
MAX_SEQ_LENGTH = 40
BUFFER_SIZE = 20000
BATCH_SIZE = 64

def load_tokenizers():
    """Loads English-German tokenizer pair from tfds."""
    tokenizer_en = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
        (en.numpy() for en, _ in tfds.as_numpy(tfds.load('ted_hrlr_translate/de_to_en', split='train[:1%]'))),
        target_vocab_size=2**13
    )
    tokenizer_de = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
        (de.numpy() for _, de in tfds.as_numpy(tfds.load('ted_hrlr_translate/de_to_en', split='train[:1%]'))),
        target_vocab_size=2**13
    )
    return tokenizer_en, tokenizer_de

def encode(lang1, lang2, tokenizer_de, tokenizer_en):
    """Encodes sentence pairs into token sequences."""
    lang1 = [tokenizer_de.vocab_size] + tokenizer_de.encode(
        lang1.numpy()) + [tokenizer_de.vocab_size + 1]
    lang2 = [tokenizer_en.vocab_size] + tokenizer_en.encode(
        lang2.numpy()) + [tokenizer_en.vocab_size + 1]
    return lang1, lang2

def tf_encode(de, en, tokenizer_de, tokenizer_en):
    """Wrap encode() in tf.py_function."""
    result_de, result_en = tf.py_function(
        func=lambda d, e: encode(d, e, tokenizer_de, tokenizer_en),
        inp=[de, en],
        Tout=[tf.int64, tf.int64]
    )
    result_de.set_shape([None])
    result_en.set_shape([None])
    return result_de, result_en

def filter_max_length(de, en, max_length=MAX_SEQ_LENGTH):
    return tf.logical_and(tf.size(de) <= max_length,
                          tf.size(en) <= max_length)

def load_dataset():
    """Loads and preprocesses the dataset."""
    tokenizer_en, tokenizer_de = load_tokenizers()

    examples, metadata = tfds.load(
        'ted_hrlr_translate/de_to_en',
        with_info=True,
        as_supervised=True
    )

    train_examples, val_examples = examples['train'], examples['validation']

    train_dataset = train_examples.map(lambda de, en: tf_encode(de, en, tokenizer_de, tokenizer_en))
    train_dataset = train_dataset.filter(filter_max_length)
    train_dataset = train_dataset.cache()
    train_dataset = train_dataset.shuffle(BUFFER_SIZE).padded_batch(BATCH_SIZE, padded_shapes=([None], [None]))
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    val_dataset = val_examples.map(lambda de, en: tf_encode(de, en, tokenizer_de, tokenizer_en))
    val_dataset = val_dataset.filter(filter_max_length).padded_batch(BATCH_SIZE, padded_shapes=([None], [None]))

    return train_dataset, val_dataset, tokenizer_en, tokenizer_de
