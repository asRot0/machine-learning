import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np


class NeuralNetworkModel:
    def __init__(self, input_shape, learning_rate=0.001):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Normalization(input_shape=input_shape),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mean_squared_error')

    def train(self, X_train, y_train, X_val, y_val, epochs=100):
        self.history = self.model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, verbose=0)

    def evaluate(self, X_test, y_test):
        mse = self.model.evaluate(X_test, y_test)
        return mse

    def plot_loss(self):
        plt.plot(self.history.history['loss'], label='loss')
        plt.plot(self.history.history['val_loss'], label='val_loss')
        plt.xlabel('Epochs')
        plt.ylabel('MSE')
        plt.grid(True)
        plt.legend()
        plt.show()

    def plot_predictions(self, X_train, y_train):
        plt.scatter(X_train, y_train, color="blue")
        x = tf.linspace(min(X_train), max(X_train), 100)
        plt.plot(x, self.model.predict(np.array(x).reshape(-1, 1)), label="Fit", color="red", linewidth=3)
        plt.xlabel("Features")
        plt.ylabel("Target")
        plt.title("Neural Network Fit")
        plt.show()
