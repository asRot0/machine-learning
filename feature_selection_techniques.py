"""
Feature Selection Techniques
============================
This script demonstrates various feature selection techniques in machine learning.
Feature selection helps improve model performance, reduce overfitting, and speed up training.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import (
    SelectKBest,
    f_classif,
    mutual_info_classif,
    RFE,
    SelectFromModel,
    VarianceThreshold,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ======================================
# 1. Generate a Dataset
# ======================================

X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=10,
    n_redundant=5,
    n_repeated=2,
    random_state=42,
)

feature_names = [f"Feature {i+1}" for i in range(X.shape[1])]
X = pd.DataFrame(X, columns=feature_names)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nDataset Information:")
print(f"Total Features: {X.shape[1]}")
print(f"Training Samples: {X_train.shape[0]}")
print(f"Testing Samples: {X_test.shape[0]}")

# ======================================
# 2. Variance Threshold
# ======================================

print("\n=== Variance Threshold ===")
var_thresh = VarianceThreshold(threshold=0.1)
X_train_var = var_thresh.fit_transform(X_train)
X_test_var = var_thresh.transform(X_test)

print(f"Features Remaining: {X_train_var.shape[1]} after removing low-variance features.")

# ======================================
# 3. Univariate Feature Selection
# ======================================

print("\n=== Univariate Feature Selection (ANOVA) ===")
k_best = SelectKBest(score_func=f_classif, k=10)
X_train_kbest = k_best.fit_transform(X_train, y_train)
X_test_kbest = k_best.transform(X_test)

selected_features = np.array(feature_names)[k_best.get_support()]
print(f"Selected Features: {selected_features}")

# ======================================
# 4. Mutual Information
# ======================================

print("\n=== Univariate Feature Selection (Mutual Information) ===")
mutual_info = SelectKBest(score_func=mutual_info_classif, k=10)
X_train_mi = mutual_info.fit_transform(X_train, y_train)
X_test_mi = mutual_info.transform(X_test)

selected_features_mi = np.array(feature_names)[mutual_info.get_support()]
print(f"Selected Features (MI): {selected_features_mi}")

# ======================================
# 5. Recursive Feature Elimination (RFE)
# ======================================

print("\n=== Recursive Feature Elimination (RFE) ===")
model = LogisticRegression(max_iter=1000, random_state=42)
rfe = RFE(estimator=model, n_features_to_select=10)
X_train_rfe = rfe.fit_transform(X_train, y_train)
X_test_rfe = rfe.transform(X_test)

selected_features_rfe = np.array(feature_names)[rfe.get_support()]
print(f"Selected Features (RFE): {selected_features_rfe}")

# ======================================
# 6. Feature Selection with Random Forest
# ======================================

print("\n=== Feature Selection with Random Forest ===")
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

importances = rf_model.feature_importances_
sorted_indices = np.argsort(importances)[::-1]

print("Feature Importances (Random Forest):")
for idx in sorted_indices[:10]:
    print(f"{feature_names[idx]}: {importances[idx]:.4f}")

# Selecting top 10 features based on importance
selected_rf_features = [feature_names[i] for i in sorted_indices[:10]]
print(f"Top 10 Features: {selected_rf_features}")

# ======================================
# 7. Evaluate Models with Selected Features
# ======================================

print("\n=== Model Evaluation with Selected Features ===")

# Using RFE-selected features for training
model_rfe = LogisticRegression(max_iter=1000, random_state=42)
model_rfe.fit(X_train_rfe, y_train)
y_pred_rfe = model_rfe.predict(X_test_rfe)
print(f"Accuracy with RFE-selected Features: {accuracy_score(y_test, y_pred_rfe):.4f}")

# Using Random Forest-selected features for training
X_train_rf_selected = X_train[selected_rf_features]
X_test_rf_selected = X_test[selected_rf_features]
rf_model_selected = RandomForestClassifier(random_state=42)
rf_model_selected.fit(X_train_rf_selected, y_train)
y_pred_rf = rf_model_selected.predict(X_test_rf_selected)
print(f"Accuracy with Random Forest-selected Features: {accuracy_score(y_test, y_pred_rf):.4f}")


'''
Variance Threshold:
    - Removes features with low variance that don't contribute much to the model.
Univariate Feature Selection (ANOVA & Mutual Information):
    - Selects features based on statistical tests.
    - ANOVA: Tests whether means of groups are significantly different.
    - Mutual Information: Measures dependency between features and the target.
Recursive Feature Elimination (RFE):
    - Recursively removes least important features based on model weights.
Feature Importance with Random Forest:
    - Uses the feature importances derived from Random Forest for selection.
Evaluation:
    - Compares model performance with selected features to assess the impact of feature selection.
'''