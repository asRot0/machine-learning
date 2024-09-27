import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from data_processing import DataProcessor


class NeuralNetworkModel:
    def __init__(self, num_nodes, dropout_prob, lr):
        """
        Initialize the neural network model with the given hyperparameters.

        Args:
            num_nodes (int): Number of nodes in the hidden layers.
            dropout_prob (float): Dropout probability.
            lr (float): Learning rate for the Adam optimizer.
        """
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(num_nodes, activation='relu', input_shape=(10,)),
            tf.keras.layers.Dropout(dropout_prob),
            tf.keras.layers.Dense(num_nodes, activation='relu'),
            tf.keras.layers.Dropout(dropout_prob),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                           loss='binary_crossentropy', metrics=['accuracy'])

    def fit(self, X_train, y_train, batch_size, epochs):
        """
        Train the neural network model.

        Args:
            X_train (np.array): Training features.
            y_train (np.array): Training labels.
            batch_size (int): Batch size for training.
            epochs (int): Number of epochs for training.

        Returns:
            history (tf.keras.callbacks.History): History object from the training process.
        """
        history = self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                                 validation_split=0.2, verbose=0)
        return history

    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on test data.

        Args:
            X_test (np.array): Test features.
            y_test (np.array): Test labels.

        Returns:
            str: Classification report as a string.
        """
        y_pred = (self.model.predict(X_test) > 0.5).astype(int).reshape(-1, )
        return classification_report(y_test, y_pred)

    @staticmethod
    def plot_history(history):
        """
        Plot the training and validation loss and accuracy over epochs.

        Args:
            history (tf.keras.callbacks.History): History object from training.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        ax1.plot(history.history['loss'], label='Training Loss')
        ax1.plot(history.history['val_loss'], label='Validation Loss')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)

        ax2.plot(history.history['accuracy'], label='Training Accuracy')
        ax2.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.show()


# Example usage
if __name__ == "__main__":
    # Define the data path and column names
    data_path = 'magic04.data'
    columns = ["fLength", "fWidth", "fSize", "fConc", "fConc1", "fAsym", "fM3Long", "fM3Trans", "fAlpha", "fDist", "class"]

    # Instantiate the data processor
    processor = DataProcessor(data_path, columns, "class")

    # Split the dataset into training, validation, and test sets
    train, valid, test = processor.split_data()

    # Preprocess and scale the data, with oversampling if needed
    X_train, y_train = processor.scale_and_resample(train, oversample=True)
    X_valid, y_valid = processor.scale_and_resample(valid, oversample=False)
    X_test, y_test = processor.scale_and_resample(test, oversample=False)

    least_val_loss = float('inf')
    best_model = None
    epochs = 100

    # Iterate over different hyperparameter combinations
    for num_nodes in [16, 32, 64]:
        for dropout_prob in [0, 0.2]:
            for lr in [0.01, 0.005, 0.001]:
                for batch_size in [32, 64, 128]:
                    print(
                        f"Training with {num_nodes} nodes, dropout {dropout_prob}, learning rate {lr}, batch size {batch_size}")
                    model = NeuralNetworkModel(num_nodes, dropout_prob, lr)
                    history = model.fit(X_train, y_train, batch_size, epochs)
                    model.plot_history(history)

                    val_loss = model.model.evaluate(X_valid, y_valid, verbose=0)[0]
                    if val_loss < least_val_loss:
                        least_val_loss = val_loss
                        best_model = model

    # Evaluate the best model on test data
    print(best_model.evaluate(X_test, y_test))
