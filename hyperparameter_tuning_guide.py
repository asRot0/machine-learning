"""
Hyperparameter Tuning in Machine Learning
=========================================
This script demonstrates various techniques for hyperparameter tuning
in machine learning, including Grid Search, Random Search, and Bayesian Optimization.
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from scipy.stats import randint
from skopt import BayesSearchCV

# Load Dataset
data = load_iris()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ===================================
# 1. Grid Search for Hyperparameter Tuning
# ===================================

print("\n=== Grid Search ===")
param_grid = {
    "n_estimators": [50, 100, 150],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5, 10]
}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=3,
    scoring="accuracy",
    verbose=1
)

grid_search.fit(X_train, y_train)
best_grid_model = grid_search.best_estimator_

print("Best Parameters from Grid Search:", grid_search.best_params_)
y_pred_grid = best_grid_model.predict(X_test)
print("Grid Search Accuracy:", accuracy_score(y_test, y_pred_grid))

# =====================================
# 2. Random Search for Hyperparameter Tuning
# =====================================

print("\n=== Random Search ===")
param_dist = {
    "n_estimators": randint(50, 200),
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": randint(2, 20)
}

random_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_distributions=param_dist,
    n_iter=10,
    cv=3,
    scoring="accuracy",
    random_state=42,
    verbose=1
)

random_search.fit(X_train, y_train)
best_random_model = random_search.best_estimator_

print("Best Parameters from Random Search:", random_search.best_params_)
y_pred_random = best_random_model.predict(X_test)
print("Random Search Accuracy:", accuracy_score(y_test, y_pred_random))

# ======================================
# 3. Bayesian Optimization for Tuning
# ======================================

print("\n=== Bayesian Optimization ===")
bayes_search = BayesSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    search_spaces=param_grid,
    n_iter=10,
    cv=3,
    scoring="accuracy",
    verbose=1
)

bayes_search.fit(X_train, y_train)
best_bayes_model = bayes_search.best_estimator_

print("Best Parameters from Bayesian Optimization:", bayes_search.best_params_)
y_pred_bayes = best_bayes_model.predict(X_test)
print("Bayesian Optimization Accuracy:", accuracy_score(y_test, y_pred_bayes))

# ======================================
# 4. Tuning for SVM Classifier (Example)
# ======================================

print("\n=== Hyperparameter Tuning for SVM ===")
svm_param_grid = {
    "C": [0.1, 1, 10],
    "gamma": [0.1, 1, 10],
    "kernel": ["rbf", "linear"]
}

svm_grid_search = GridSearchCV(
    estimator=SVC(),
    param_grid=svm_param_grid,
    cv=3,
    scoring="accuracy",
    verbose=1
)

svm_grid_search.fit(X_train, y_train)
best_svm_model = svm_grid_search.best_estimator_

print("Best Parameters for SVM:", svm_grid_search.best_params_)
y_pred_svm = best_svm_model.predict(X_test)
print("SVM Grid Search Accuracy:", accuracy_score(y_test, y_pred_svm))
print("\nClassification Report for SVM:\n", classification_report(y_test, y_pred_svm))

# ======================================
# 5. Automating Hyperparameter Tuning
# ======================================

# Using libraries like Optuna or HyperOpt for automated and efficient hyperparameter tuning.
try:
    import optuna
except ImportError:
    print("\n[Optional] Consider using Optuna for advanced hyperparameter tuning.\nTo install: pip install optuna")

# ======================================
# Summary of Methods and Results
# ======================================

print("\n=== Summary of Results ===")
print("Grid Search Accuracy:", accuracy_score(y_test, y_pred_grid))
print("Random Search Accuracy:", accuracy_score(y_test, y_pred_random))
print("Bayesian Optimization Accuracy:", accuracy_score(y_test, y_pred_bayes))
print("SVM Grid Search Accuracy:", accuracy_score(y_test, y_pred_svm))


'''
Grid Search:
    - Exhaustive search over a predefined hyperparameter grid.
    - Advantage: Finds the best combination of hyperparameters.
    - Disadvantage: Computationally expensive.
Random Search:
    - Randomly samples hyperparameters from a specified range.
    - Advantage: Faster than grid search, effective when the hyperparameter space is large.
    - Disadvantage: May miss the optimal values due to randomness.
Bayesian Optimization:
    - Uses probabilistic models to select hyperparameters more efficiently.
    - Advantage: Balances exploration and exploitation for tuning.
    - Disadvantage: Requires additional libraries like skopt or optuna.
Hyperparameter Tuning for SVM:
    - Tuning C, gamma, and kernel to optimize support vector machines.
Automating Tuning:
    - Mentions libraries like Optuna for more advanced automation.
'''