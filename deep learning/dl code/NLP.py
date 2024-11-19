# NLP with BERT for Text Classification

from transformers import TFBertForSequenceClassification, BertTokenizer
import tensorflow as tf

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = TFBertForSequenceClassification.from_pretrained("bert-base-uncased")

# Prepare dataset
sentences = ["I love machine learning.", "This is a bad example."]
labels = [1, 0]

# Tokenization and encoding
inputs = tokenizer(sentences, return_tensors="tf", padding=True, truncation=True)
labels = tf.constant(labels)

# Compile and train model
optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)
model.compile(optimizer=optimizer, loss=model.compute_loss)
model.fit(inputs, labels, epochs=2)

# Explanation:
# This code uses a pre-trained BERT model for text classification.
# BERT's transformer-based architecture is well-suited for NLP tasks like text classification.
