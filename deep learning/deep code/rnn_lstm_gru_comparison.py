import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, LSTM, GRU, Dense
from tensorflow.keras.optimizers import Adam

# Generate synthetic sequential data
def generate_data(samples=1000, timesteps=10, features=5):
    X = np.random.randn(samples, timesteps, features)
    y = np.random.randint(0, 2, size=(samples, 1))  # Binary classification
    return X, y

# Create RNN model
def create_rnn_model(input_shape):
    model = Sequential([
        SimpleRNN(32, activation='relu', return_sequences=False, input_shape=input_shape),
        Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    return model

# Create LSTM model
def create_lstm_model(input_shape):
    model = Sequential([
        LSTM(32, activation='relu', return_sequences=False, input_shape=input_shape),
        Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    return model

# Create GRU model
def create_gru_model(input_shape):
    model = Sequential([
        GRU(32, activation='relu', return_sequences=False, input_shape=input_shape),
        Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    return model

# Load data
X, y = generate_data()
input_shape = X.shape[1:]  # (timesteps, features)

# Train and evaluate models
for model_name, create_model in zip(["RNN", "LSTM", "GRU"],
                                    [create_rnn_model, create_lstm_model, create_gru_model]):
    print(f"\nTraining {model_name} model...")
    model = create_model(input_shape)
    model.fit(X, y, epochs=10, batch_size=32, verbose=1, validation_split=0.2)
    loss, acc = model.evaluate(X, y, verbose=0)
    print(f"{model_name} Accuracy: {acc:.4f}\n")
