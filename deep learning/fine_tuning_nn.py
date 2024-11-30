# Fine-Tuning Neural Network Hyperparameters with RandomizedSearchCV

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, RandomizedSearchCV
import tensorflow as tf
from tensorflow import keras
from scipy.stats import reciprocal
import numpy as np

'''
This script demonstrates:
1. Building a customizable neural network for regression using TensorFlow/Keras.
2. Wrapping the Keras model for compatibility with Scikit-Learn.
3. Performing hyperparameter tuning using RandomizedSearchCV to find the best
   combination of number of hidden layers, neurons, and learning rate.
'''


# Function to construct a customizable neural network
def build_model(n_hidden=1, n_neurons=30, learning_rate=3e-3, input_shape=(8,)):
    '''
    Build a Sequential Keras model for regression tasks.

    Parameters:
    - n_hidden: Number of hidden layers.
    - n_neurons: Number of neurons per hidden layer.
    - learning_rate: Learning rate for the SGD optimizer.
    - input_shape: Shape of the input features.

    Returns:
    - A compiled Keras Sequential model with specified architecture and parameters.
    '''
    model = keras.models.Sequential()
    options = {'input_shape': input_shape}

    # Adding hidden layers based on n_hidden
    for layer in range(n_hidden):
        model.add(keras.layers.Dense(n_neurons, activation='relu', **options))
        options = {}  # Only specify input_shape for the first layer

    # Adding the output layer
    model.add(keras.layers.Dense(1, **options))  # Regression has 1 output neuron

    # Compiling the model with SGD optimizer
    optimizer = keras.optimizers.SGD(learning_rate=learning_rate)
    model.compile(loss='mse', optimizer=optimizer)  # MSE for regression tasks
    return model


# Load California Housing dataset
housing = fetch_california_housing()
X_train_full, X_test, y_train_full, y_test = train_test_split(housing.data, housing.target)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full)

# Take a small subset of the test set for demonstration
X_new = X_test[:3]

# Wrap the Keras model with Scikit-Learn wrapper
keras_reg = keras.wrappers.scikit_learn.KerasRegressor(build_model)

# Train the model and enable early stopping to prevent overfitting
keras_reg.fit(X_train, y_train, epochs=100,
              validation_data=(X_valid, y_valid),
              callbacks=[keras.callbacks.EarlyStopping(patience=10)])

# Evaluate the model's performance
mse_test = keras_reg.score(X_test, y_test)  # MSE on test data
y_pred = keras_reg.predict(X_new)  # Predictions for a few samples
print("Mean Squared Error on Test Data:", mse_test)
print("Predictions for new data:", y_pred)

# Define hyperparameter search space
param_distribs = {
    'n_hidden': [0, 1, 2, 3],  # Number of hidden layers
    'n_neurons': np.arange(1, 100),  # Neurons in each layer
    'learning_rate': reciprocal(3e-4, 3e-2),  # Log-uniform sampling for learning rates
}

# Perform Randomized Search for hyperparameter optimization
rnd_search_cv = RandomizedSearchCV(keras_reg, param_distribs, n_iter=10, cv=3)

# Train with hyperparameter tuning
rnd_search_cv.fit(X_train, y_train, epochs=100,
                  validation_data=(X_valid, y_valid),
                  callbacks=[keras.callbacks.EarlyStopping(patience=10)])

'''
>>> rnd_search_cv.best_params_
{'learning_rate': 0.0033625641252688094, 'n_hidden': 2, 'n_neurons': 42}
>>> rnd_search_cv.best_score_
-0.3189529188278931
>>> model = rnd_search_cv.best_estimator_.model
'''

# Extract best parameters and score
print("Best Parameters:", rnd_search_cv.best_params_)
print("Best Cross-Validated Score:", rnd_search_cv.best_score_)

# Retrieve the best model from the search
best_model = rnd_search_cv.best_estimator_.model
print("Best Model Architecture Summary:")
best_model.summary()
