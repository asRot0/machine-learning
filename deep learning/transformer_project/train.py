import tensorflow as tf
from .dataset_loader import load_dataset
from .models.transformer import Transformer

# Hyperparameters
NUM_LAYERS = 4
EMBED_SIZE = 128
NUM_HEADS = 8
FF_EXPANSION = 4
DROPOUT_RATE = 0.1
EPOCHS = 10

# Load data
train_dataset, val_dataset, tokenizer_de, tokenizer_en = load_dataset()
input_vocab_size = tokenizer_de.vocab_size + 2  # +2 for <start>, <end>
target_vocab_size = tokenizer_en.vocab_size + 2

# Create model
model = Transformer(
    num_layers=NUM_LAYERS,
    embed_size=EMBED_SIZE,
    num_heads=NUM_HEADS,
    ff_expansion=FF_EXPANSION,
    input_vocab_size=input_vocab_size,
    target_vocab_size=target_vocab_size,
    max_seq_length=40,
    dropout_rate=DROPOUT_RATE
)

# Define loss & accuracy
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

def loss_function(y_true, y_pred):
    mask = tf.math.logical_not(tf.math.equal(y_true, 0))  # Padding mask
    loss_ = loss_object(y_true, y_pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask  # Zero out padding loss

    return tf.reduce_sum(loss_) / tf.reduce_sum(mask)

def accuracy_function(y_true, y_pred):
    predictions = tf.argmax(y_pred, axis=2)
    mask = tf.math.logical_not(tf.math.equal(y_true, 0))
    matches = tf.cast(tf.equal(y_true, predictions), dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(matches * mask) / tf.reduce_sum(mask)

# Compile model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss=loss_function,
    metrics=[accuracy_function]
)

# Prepare inputs and targets
def split_inputs_and_targets(de, en):
    """Splits target input and output for decoder."""
    decoder_input = en[:, :-1]
    target_output = en[:, 1:]
    return (de, decoder_input), target_output

train_dataset = train_dataset.map(split_inputs_and_targets)
val_dataset = val_dataset.map(split_inputs_and_targets)

# Train the model
model.fit(train_dataset, validation_data=val_dataset, epochs=EPOCHS)
