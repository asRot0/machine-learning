import numpy as np
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, explained_variance_score, r2_score,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
)
import tensorflow as tf

# ---- Introduction ----
# This script explains various loss functions and evaluation metrics, their use cases,
# and why and when to use them. Includes both basic and advanced methods for regression
# and classification problems.

# ========================================================
# 1. LOSS FUNCTIONS FOR REGRESSION
# ========================================================
print("\n--- Regression Loss Functions ---\n")

# 1. Regression: Common Loss Functions
#    - Mean Squared Error (MSE)
#    - Mean Absolute Error (MAE)

# Simulated Regression Data
y_true_reg = np.array([3.0, -0.5, 2.0, 7.0])
y_pred_reg = np.array([2.5, 0.0, 2.0, 8.0])

# 1.1 Mean Squared Error (MSE)
# Purpose: Penalizes large errors more than smaller ones. Sensitive to outliers.
# Penalizes large errors more heavily. Use when outliers are important to consider.
mse = mean_squared_error(y_true_reg, y_pred_reg)
print(f"MSE (Mean Squared Error): {mse:.2f}")

# 1.2 Mean Absolute Error (MAE)
# Purpose: Penalizes all errors equally. Less sensitive to outliers.
# Penalizes errors linearly. Less sensitive to outliers.
mae = mean_absolute_error(y_true_reg, y_pred_reg)
print(f"MAE (Mean Absolute Error): {mae:.2f}")

# 1.3 Huber Loss
# Combines MSE and MAE to handle outliers better. Use when data has noise or outliers.
delta = 1.0
huber_loss = np.mean([
    0.5 * (y - y_pred)**2 if abs(y - y_pred) <= delta else delta * abs(y - y_pred) - 0.5 * delta**2
    for y, y_pred in zip(y_true_reg, y_pred_reg)
])
print(f"Huber Loss: {huber_loss:.2f}")

# 1.4 Log-Cosh Loss
# Similar to MAE but differentiable everywhere. Use when small outliers are acceptable.
log_cosh_loss = np.mean(np.log(np.cosh(y_pred_reg - y_true_reg)))
print(f"Log-Cosh Loss: {log_cosh_loss:.2f}")

# 1.5 R^2 Score
# Measures goodness of fit. Values closer to 1 indicate better fit.
r2 = r2_score(y_true_reg, y_pred_reg)
print(f"R² Score: {r2:.2f}")

# ========================================================
# 2. LOSS FUNCTIONS FOR CLASSIFICATION
# ========================================================
print("\n--- Classification Loss Functions ---\n")

# 2. Classification: Common Loss Functions
#    - Binary Cross-Entropy (BCE)
#    - Categorical Cross-Entropy (CCE)

# Simulated Classification Data
y_true_bin = np.array([1, 0, 1, 1])  # Binary labels
y_pred_bin = np.array([0.9, 0.2, 0.8, 0.7])  # Predicted probabilities
y_true_cat = np.array([2, 0, 1])  # Labels for 3 classes
y_pred_cat = np.array([
    [0.1, 0.7, 0.2],
    [0.8, 0.1, 0.1],
    [0.2, 0.5, 0.3]
])  # Predicted probabilities for 3 classes

# 2.1 Binary Cross-Entropy (BCE)
# Suitable for binary classification tasks (e.g., spam detection).
bce = tf.keras.losses.BinaryCrossentropy()(y_true_bin, y_pred_bin).numpy()
print(f"Binary Cross-Entropy (BCE): {bce:.2f}")

# 2.2 Categorical Cross-Entropy (CCE)
# Used for multi-class classification tasks.
cce = tf.keras.losses.CategoricalCrossentropy()(tf.one_hot(y_true_cat, depth=3), y_pred_cat).numpy()
print(f"Categorical Cross-Entropy (CCE): {cce:.2f}")

# 2.3 Sparse Categorical Cross-Entropy (SCCE)
# Similar to CCE but works with integer labels instead of one-hot encoding.
scce = tf.keras.losses.SparseCategoricalCrossentropy()(y_true_cat, y_pred_cat).numpy()
print(f"Sparse Categorical Cross-Entropy (SCCE): {scce:.2f}")

# ========================================================
# 3. EVALUATION METRICS FOR CLASSIFICATION
# ========================================================
print("\n--- Classification Evaluation Metrics ---\n")

# 3. Evaluation Metrics: Classification
#    - Accuracy
#    - F1 Score
#    - Confusion Matrix

# Convert predicted probabilities to class labels
y_pred_classes = np.argmax(y_pred_cat, axis=1)

# 3.1 Accuracy
# Simple metric for balanced datasets.
# Purpose: Measures the proportion of correctly classified instances.
accuracy = accuracy_score(y_true_cat, y_pred_classes)
print(f"Accuracy: {accuracy:.2f}")

# 3.2 Precision
# Measures true positives out of predicted positives. Use when false positives are costly.
precision = precision_score(y_true_cat, y_pred_classes, average='weighted', zero_division=1)
print(f"Precision: {precision:.2f}")

# 3.3 Recall
# Measures true positives out of actual positives. Use when false negatives are costly.
recall = recall_score(y_true_cat, y_pred_classes, average='weighted', zero_division=1)
print(f"Recall: {recall:.2f}")

# 3.4 F1 Score
# Harmonic mean of precision and recall. Useful for imbalanced datasets.
# Purpose: Balances precision and recall. Used for imbalanced datasets.
f1 = f1_score(y_true_cat, y_pred_classes, average='weighted')
print(f"F1 Score: {f1:.2f}")

# 3.5 ROC-AUC Score
# Evaluates binary classifiers over various threshold settings.
roc_auc = roc_auc_score(y_true_bin, y_pred_bin)
print(f"ROC-AUC Score: {roc_auc:.2f}")

# 3.6 Confusion Matrix
# Provides a detailed breakdown of true/false positives and negatives.
# Purpose: Shows the counts of true positives, false positives, true negatives, and false negatives.
conf_matrix = confusion_matrix(y_true_cat, y_pred_classes)
print(f"Confusion Matrix:\n{conf_matrix}")

# Classification:
# - Precision: Measures how many of the predicted positives are actual positives.
# - Recall: Measures how many of the actual positives were correctly predicted.
# Example:
precision = np.diag(conf_matrix) / np.sum(conf_matrix, axis=0)
recall = np.diag(conf_matrix) / np.sum(conf_matrix, axis=1)
print(f"Precision: {precision}")
print(f"Recall: {recall}")

# ========================================================
# 4. WHEN TO USE WHICH LOSS FUNCTION OR METRIC?
# ========================================================

print("\n--- Summary of Use Cases ---\n")
print("""
1. Regression:
   - Use MSE when large errors need heavy penalization.
   - Use MAE when outliers should have less impact.
   - Use Huber or Log-Cosh Loss for noisy datasets with some outliers.
   - Huber Loss: Combines MSE and MAE for robustness to outliers.
   - Log-Cosh Loss: Similar to MAE but differentiable everywhere.

2. Binary Classification:
   - Use Binary Cross-Entropy for binary classification.
   - Use ROC-AUC Score to measure performance across thresholds.

3. Multi-Class Classification:
   - Use Categorical Cross-Entropy for one-hot encoded labels.
   - Use Sparse Categorical Cross-Entropy for integer labels.
   - Use F1 Score for imbalanced datasets.

4. Evaluation Metrics:
   - Accuracy: Use for balanced datasets.
   - Precision: Use when false positives are costly.
   - Recall: Use when false negatives are costly.
   - F1 Score: Use for imbalanced datasets.
   - ROC-AUC: Use for evaluating binary classifiers.
""")
