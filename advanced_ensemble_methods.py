"""
Advanced Ensemble Techniques
============================
This script explores advanced ensemble techniques in machine learning,
including Stacking, Blending, Bagging, and Boosting methods.
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# ======================================
# 1. Generate a Dataset
# ======================================

X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ======================================
# 2. Voting Classifier (Hard & Soft Voting)
# ======================================

print("\n=== Voting Classifier ===")
rf_clf = RandomForestClassifier(random_state=42)
svc_clf = SVC(probability=True, random_state=42)
log_reg = LogisticRegression(random_state=42)

# Hard Voting
hard_voting_clf = VotingClassifier(estimators=[
    ('rf', rf_clf),
    ('svc', svc_clf),
    ('lr', log_reg)
], voting='hard')

hard_voting_clf.fit(X_train, y_train)
y_pred_hard = hard_voting_clf.predict(X_test)
print("Hard Voting Accuracy:", accuracy_score(y_test, y_pred_hard))

# Soft Voting
soft_voting_clf = VotingClassifier(estimators=[
    ('rf', rf_clf),
    ('svc', svc_clf),
    ('lr', log_reg)
], voting='soft')

soft_voting_clf.fit(X_train, y_train)
y_pred_soft = soft_voting_clf.predict(X_test)
print("Soft Voting Accuracy:", accuracy_score(y_test, y_pred_soft))

# ======================================
# 3. Bagging Classifier
# ======================================

print("\n=== Bagging Classifier ===")
bagging_clf = BaggingClassifier(
    base_estimator=LogisticRegression(),
    n_estimators=50,
    max_samples=0.8,
    max_features=1.0,
    bootstrap=True,
    random_state=42
)

bagging_clf.fit(X_train, y_train)
y_pred_bagging = bagging_clf.predict(X_test)
print("Bagging Classifier Accuracy:", accuracy_score(y_test, y_pred_bagging))

# ======================================
# 4. Boosting (Gradient Boosting)
# ======================================

print("\n=== Gradient Boosting ===")
gb_clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
gb_clf.fit(X_train, y_train)
y_pred_gb = gb_clf.predict(X_test)
print("Gradient Boosting Accuracy:", accuracy_score(y_test, y_pred_gb))

# ======================================
# 5. Stacking Classifier
# ======================================

print("\n=== Stacking Classifier ===")
stacking_clf = LogisticRegression()

from sklearn.ensemble import StackingClassifier
stacking_model = StackingClassifier(
    estimators=[
        ('rf', rf_clf),
        ('svc', svc_clf)
    ],
    final_estimator=stacking_clf
)

stacking_model.fit(X_train, y_train)
y_pred_stacking = stacking_model.predict(X_test)
print("Stacking Classifier Accuracy:", accuracy_score(y_test, y_pred_stacking))

# ======================================
# 6. Comparison of Ensemble Methods
# ======================================

print("\n=== Comparison of Ensemble Methods ===")
print(f"Hard Voting Accuracy: {accuracy_score(y_test, y_pred_hard):.3f}")
print(f"Soft Voting Accuracy: {accuracy_score(y_test, y_pred_soft):.3f}")
print(f"Bagging Classifier Accuracy: {accuracy_score(y_test, y_pred_bagging):.3f}")
print(f"Gradient Boosting Accuracy: {accuracy_score(y_test, y_pred_gb):.3f}")
print(f"Stacking Classifier Accuracy: {accuracy_score(y_test, y_pred_stacking):.3f}")


'''
Voting Classifier:
    - Combines predictions from multiple models by majority voting (hard) or by averaging predicted probabilities (soft).
    - Advantage: Works well if base models are diverse.
Bagging Classifier:
    - Trains multiple models on different subsets of the data (with replacement).
    - Reduces variance and prevents overfitting (e.g., Random Forest is a Bagging method).
Boosting Classifier:
    - Sequentially trains models, where each model focuses on correcting the errors of the previous one.
    - Gradient Boosting is a popular implementation.
Stacking Classifier:
    - Combines predictions from base models using a meta-model.
    - Advantage: Often outperforms individual models by leveraging their strengths.
Comparison of Methods:
    - Evaluates the accuracy of each ensemble method to determine the most effective one for the dataset.
'''