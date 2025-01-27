"""
Hyperparameter Tuning
=====================
This script demonstrates various hyperparameter tuning methods in machine learning, including:
1. Grid Search
2. Randomized Search
3. Bayesian Optimization
4. Hyperband with Keras Tuner
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import keras_tuner as kt
from tensorflow import keras

# ======================================
# 1. Generate a Dataset
# ======================================

X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=15,
    n_redundant=3,
    random_state=42,
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nDataset Information:")
print(f"Training Samples: {X_train.shape[0]}")
print(f"Testing Samples: {X_test.shape[0]}")

# ======================================
# 2. Grid Search
# ======================================

print("\n=== Grid Search ===")
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
}
rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring="accuracy", n_jobs=-1)
grid_search.fit(X_train, y_train)

print(f"Best Parameters (Grid Search): {grid_search.best_params_}")
best_rf_grid = grid_search.best_estimator_
y_pred_grid = best_rf_grid.predict(X_test)
print(f"Accuracy (Grid Search): {accuracy_score(y_test, y_pred_grid):.4f}")

# ======================================
# 3. Randomized Search
# ======================================

print("\n=== Randomized Search ===")
param_distributions = {
    "n_estimators": [int(x) for x in np.linspace(50, 200, 10)],
    "max_depth": [None] + list(range(10, 31, 5)),
    "min_samples_split": [2, 5, 10],
}
random_search = RandomizedSearchCV(
    estimator=rf, param_distributions=param_distributions, n_iter=50, cv=3, scoring="accuracy", n_jobs=-1, random_state=42
)
random_search.fit(X_train, y_train)

print(f"Best Parameters (Randomized Search): {random_search.best_params_}")
best_rf_random = random_search.best_estimator_
y_pred_random = best_rf_random.predict(X_test)
print(f"Accuracy (Randomized Search): {accuracy_score(y_test, y_pred_random):.4f}")

# ======================================
# 4. Bayesian Optimization with Keras Tuner
# ======================================

print("\n=== Bayesian Optimization with Keras Tuner ===")

def build_model(hp):
    model = keras.Sequential([
        keras.layers.Dense(units=hp.Int("units", min_value=16, max_value=128, step=16), activation="relu"),
        keras.layers.Dense(1, activation="sigmoid")
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model

tuner = kt.BayesianOptimization(
    hypermodel=build_model,
    objective="val_accuracy",
    max_trials=10,
    directory="my_dir",
    project_name="bayesian_tuning"
)

# Preparing the data for Keras
X_train_nn = X_train.astype(np.float32)
X_test_nn = X_test.astype(np.float32)
y_train_nn = y_train.astype(np.float32)
y_test_nn = y_test.astype(np.float32)

tuner.search(X_train_nn, y_train_nn, validation_split=0.2, epochs=10, verbose=1)

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print(f"Best Hyperparameters (Bayesian Optimization): {best_hps.values}")

best_model = tuner.hypermodel.build(best_hps)
history = best_model.fit(X_train_nn, y_train_nn, validation_split=0.2, epochs=10, verbose=1)
test_loss, test_acc = best_model.evaluate(X_test_nn, y_test_nn, verbose=0)
print(f"Accuracy (Bayesian Optimization): {test_acc:.4f}")

# ======================================
# 5. Summary of Results
# ======================================

print("\n=== Summary ===")
print(f"Accuracy (Grid Search): {accuracy_score(y_test, y_pred_grid):.4f}")
print(f"Accuracy (Randomized Search): {accuracy_score(y_test, y_pred_random):.4f}")
print(f"Accuracy (Bayesian Optimization): {test_acc:.4f}")


'''
Grid Search:
    - Exhaustively searches over a specified parameter grid.
    - Best for smaller parameter spaces where computational cost is manageable.
Randomized Search:
    - Randomly samples parameter combinations.
    - Efficient for larger parameter spaces, though it may not find the global best solution.
Bayesian Optimization:
    - Uses Keras Tuner to optimize hyperparameters based on past trials.
    - Balances exploration (trying new hyperparameters) and exploitation (refining promising ones).
Evaluation:
    - Each method’s performance is tested on the dataset, and results are compared.
'''