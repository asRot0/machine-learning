"""
Model Selection and Hyperparameter Tuning
==========================================
This script demonstrates how to select the best model for your dataset
and optimize hyperparameters using GridSearchCV and RandomizedSearchCV.
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import numpy as np

# Load Example Dataset
data = load_iris()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training Samples: {X_train.shape[0]}, Test Samples: {X_test.shape[0]}")

# 1. Model Selection: Comparing Different Models
models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "SVM": SVC(random_state=42)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    print(f"{name} Accuracy: {acc:.4f}")

# 2. Hyperparameter Tuning with GridSearchCV
param_grid_rf = {
    'n_estimators': [50, 100, 150],
    'max_depth': [3, 5, 10, None],
    'min_samples_split': [2, 5, 10]
}

print("\nPerforming GridSearchCV for Random Forest...")
grid_search_rf = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid_rf,
    cv=3,
    scoring='accuracy',
    n_jobs=-1
)
grid_search_rf.fit(X_train, y_train)

print("Best Parameters for Random Forest:", grid_search_rf.best_params_)
print("Best Cross-Validated Accuracy (RF):", grid_search_rf.best_score_)

# 3. Hyperparameter Tuning with RandomizedSearchCV
param_dist_svc = {
    'C': np.logspace(-3, 3, 7),
    'gamma': np.logspace(-3, 3, 7),
    'kernel': ['rbf', 'poly', 'sigmoid']
}

print("\nPerforming RandomizedSearchCV for SVM...")
random_search_svc = RandomizedSearchCV(
    SVC(random_state=42),
    param_distributions=param_dist_svc,
    n_iter=20,
    cv=3,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42
)
random_search_svc.fit(X_train, y_train)

print("Best Parameters for SVM:", random_search_svc.best_params_)
print("Best Cross-Validated Accuracy (SVM):", random_search_svc.best_score_)

# 4. Final Evaluation with Best Models
print("\nEvaluating Best Models on Test Data...")

best_rf = grid_search_rf.best_estimator_
best_svc = random_search_svc.best_estimator_

rf_test_acc = accuracy_score(y_test, best_rf.predict(X_test))
svc_test_acc = accuracy_score(y_test, best_svc.predict(X_test))

print(f"Final Test Accuracy (Random Forest): {rf_test_acc:.4f}")
print(f"Final Test Accuracy (SVM): {svc_test_acc:.4f}")

# 5. Choosing the Best Model
if rf_test_acc > svc_test_acc:
    print("\nThe Best Model for This Dataset is Random Forest.")
else:
    print("\nThe Best Model for This Dataset is SVM.")

'''
Model Selection: Compares two popular models, Random Forest and SVM, to find the one with the best baseline accuracy.
GridSearchCV: Systematic search over a predefined grid of hyperparameters for Random Forest.
RandomizedSearchCV: Randomized search over a wide range of hyperparameters for SVM.
Final Evaluation: Uses the best hyperparameters to evaluate performance on unseen test data.
Best Model Selection: Selects the model with the highest test accuracy.
'''