# ensemble_learning_comparison.py

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.ensemble import (
    VotingClassifier, BaggingClassifier, RandomForestClassifier,
    AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, StackingClassifier
)
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

# Load the California housing dataset
data = fetch_california_housing()
X = data.data
y = data.target > 2.5  # Binary classification: High/Low priced houses

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 1. Voting Classifier with soft voting
voting_clf = VotingClassifier(
    estimators=[('lr', LogisticRegression(max_iter=1000)),
                ('svc', SVC(probability=True, gamma='auto')),
                ('dt', DecisionTreeClassifier())],
    voting='soft'
)
voting_clf.fit(X_train, y_train)

# 2. Bagging Classifier with Decision Tree
bagging_clf = BaggingClassifier(
    DecisionTreeClassifier(), n_estimators=100, bootstrap=True, random_state=42
)
bagging_clf.fit(X_train, y_train)

# 3. Out-of-Bag Evaluation for Bagging
bagging_clf_oob = BaggingClassifier(
    DecisionTreeClassifier(), n_estimators=100, bootstrap=True, oob_score=True, random_state=42
)
bagging_clf_oob.fit(X_train, y_train)
print("Bagging Classifier OOB Score:", bagging_clf_oob.oob_score_)

# 4. Random Forest for Feature Importance
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_train)
print("Random Forest Feature Importances:", rf_clf.feature_importances_)

# 5. Extra-Trees Classifier
extra_trees_clf = ExtraTreesClassifier(n_estimators=100, random_state=42)
extra_trees_clf.fit(X_train, y_train)

# 6. AdaBoost with Decision Stumps
ada_clf = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=1), n_estimators=100, random_state=42
)
ada_clf.fit(X_train, y_train)

# 7. Gradient Boosting Classifier
gb_clf = GradientBoostingClassifier(
    n_estimators=100, learning_rate=0.1, random_state=42
)
gb_clf.fit(X_train, y_train)

# 8. Stacking Classifier with a Logistic Regression meta-learner
stacking_clf = StackingClassifier(
    estimators=[('lr', LogisticRegression(max_iter=1000)),
                ('svc', SVC(probability=True, gamma='auto')),
                ('dt', DecisionTreeClassifier())],
    final_estimator=LogisticRegression()
)
stacking_clf.fit(X_train, y_train)

# Evaluate and display model performances
models = {
    "Voting Classifier": voting_clf,
    "Bagging Classifier": bagging_clf,
    "Bagging Classifier with OOB": bagging_clf_oob,
    "Random Forest": rf_clf,
    "Extra Trees": extra_trees_clf,
    "AdaBoost": ada_clf,
    "Gradient Boosting": gb_clf,
    "Stacking Classifier": stacking_clf
}

print("\nModel Performance:")
for name, model in models.items():
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {accuracy:.4f}")
