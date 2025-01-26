"""
Evaluation Metrics and Model Validation Techniques
===================================================
This script demonstrates the use of various evaluation metrics
and model validation techniques for assessing AI/ML models.
"""

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve, classification_report
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load Example Dataset (Handwritten Digits)
data = load_digits()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training Samples: {X_train.shape[0]}, Test Samples: {X_test.shape[0]}")

# Train Logistic Regression Model
model = LogisticRegression(max_iter=10000, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)

# 1. Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# 2. Precision, Recall, and F1-Score
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

# 3. Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt='d', xticklabels=data.target_names, yticklabels=data.target_names)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()

# 4. ROC-AUC Score (for Binary Classification)
# For simplicity, we convert this to a binary classification task
binary_y = (y == 1).astype(int)
binary_y_train, binary_y_test = train_test_split(binary_y, test_size=0.2, random_state=42)

binary_model = LogisticRegression(max_iter=10000, random_state=42)
binary_model.fit(X_train, binary_y_train)
binary_y_proba = binary_model.predict_proba(X_test)[:, 1]

roc_auc = roc_auc_score(binary_y_test, binary_y_proba)
fpr, tpr, _ = roc_curve(binary_y_test, binary_y_proba)

print(f"ROC-AUC Score: {roc_auc:.4f}")

# Plot ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"Logistic Regression (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.show()

# 5. Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=data.target_names.astype(str)))

# 6. Cross-Validation
cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f"Cross-Validation Scores: {cv_scores}")
print(f"Mean CV Accuracy: {np.mean(cv_scores):.4f}")

# 7. Stratified K-Fold Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
stratified_scores = []

for train_idx, val_idx in skf.split(X, y):
    X_train_fold, X_val_fold = X[train_idx], X[val_idx]
    y_train_fold, y_val_fold = y[train_idx], y[val_idx]
    model.fit(X_train_fold, y_train_fold)
    fold_acc = model.score(X_val_fold, y_val_fold)
    stratified_scores.append(fold_acc)

print(f"Stratified K-Fold Scores: {stratified_scores}")
print(f"Mean Stratified K-Fold Accuracy: {np.mean(stratified_scores):.4f}")

# 8. Comparing Models with Validation
models = {
    "Logistic Regression": LogisticRegression(max_iter=10000, random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42)
}

for name, model in models.items():
    cv_score = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    print(f"{name} Mean CV Accuracy: {np.mean(cv_score):.4f}")


'''
Accuracy: Measures the percentage of correct predictions.
Precision, Recall, F1-Score: Key metrics for imbalanced datasets:
    - Precision: True positives / (True positives + False positives).
    - Recall: True positives / (True positives + False negatives).
    - F1-Score: Harmonic mean of Precision and Recall.
Confusion Matrix: Provides a matrix representation of True/False Positives/Negatives.
ROC-AUC Score: Useful for evaluating binary classification models, independent of threshold selection.
Classification Report: Summarizes key metrics for each class.
Cross-Validation: Evaluates model performance on multiple dataset splits.
Stratified K-Fold: Ensures class proportions are maintained in each fold.
Model Comparison: Uses CV scores to compare multiple models.
'''