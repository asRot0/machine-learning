# RNN for Sequence Prediction (rnn_text.py)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.datasets import imdb

# Load and preprocess dataset
max_features = 10000
maxlen = 500
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=max_features)
X_train, X_test = pad_sequences(X_train, maxlen=maxlen), pad_sequences(X_test, maxlen=maxlen)

# Build RNN model
model = Sequential([
    Embedding(max_features, 32, input_length=maxlen),
    SimpleRNN(32, return_sequences=False),
    Dense(1, activation='sigmoid')
])

# Compile and train
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=5, batch_size=64, verbose=1)
print("RNN Accuracy:", model.evaluate(X_test, y_test, verbose=0)[1])


'''
Explanation:

Basic RNN model for text sentiment analysis using the IMDB dataset.
Uses an Embedding layer to convert word indices into dense vectors and a sigmoid output for binary classification.
'''