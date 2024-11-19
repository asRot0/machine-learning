# Recurrent Neural Networks (RNNs)
# Basic RNN for Time-Series Prediction (LSTM for Sequential Data)

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# Generate synthetic sequential data
time_steps = np.linspace(0, 100, 500)
data = np.sin(time_steps)  # sine wave data for time series

# Prepare data for LSTM
window_size = 10
X = [data[i:i+window_size] for i in range(len(data) - window_size)]
y = data[window_size:]

# Scale and reshape data
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
X = np.array(X).reshape(-1, window_size, 1)
y = scaler.transform(y.reshape(-1, 1))

# Build LSTM model for time-series prediction
model = Sequential([
    LSTM(50, activation='relu', input_shape=(window_size, 1)),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=20, batch_size=32, verbose=1)

# Explanation:
# LSTM layers are used here to learn dependencies in sequential data.
# This model is trained to predict a time-series trend based on previous data points.
