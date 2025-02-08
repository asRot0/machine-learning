import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import (
    RandomForestClassifier, BaggingClassifier, VotingClassifier,
    StackingClassifier, ExtraTreesClassifier, GradientBoostingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load dataset (example: Iris)
from sklearn.datasets import load_iris
data = load_iris()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 1️⃣ Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# 2️⃣ Bagging Classifier
bagging = BaggingClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=50, random_state=42)

# 3️⃣ Extra Trees Classifier
extra_trees = ExtraTreesClassifier(n_estimators=100, random_state=42)

# 4️⃣ Gradient Boosting
gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)

# 5️⃣ Voting Classifier (Hard Voting)
voting = VotingClassifier(estimators=[
    ('rf', rf), ('gb', gb), ('et', extra_trees)
], voting='hard')

# 6️⃣ Stacking Classifier
stacking = StackingClassifier(estimators=[
    ('rf', rf), ('gb', gb), ('et', extra_trees)
], final_estimator=LogisticRegression())

# Fit models
models = {
    "Random Forest": rf,
    "Bagging Classifier": bagging,
    "Extra Trees": extra_trees,
    "Gradient Boosting": gb,
    "Voting Classifier": voting,
    "Stacking Classifier": stacking
}

# Train and Evaluate
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{name}: Accuracy = {accuracy:.4f}")

'''
| Algorithm            | Best For                                   | Pros                                      | Cons                                    |
|---------------------|-------------------------------------------|------------------------------------------|----------------------------------------|
| Random Forest      | General-purpose, structured data          | Reduces overfitting, easy to interpret  | Slower for large datasets              |
| Bagging Classifier | Reducing variance, small datasets         | Reduces overfitting, improves stability | Not ideal for high-bias models         |
| Stacking           | Combining multiple models for better accuracy | Uses multiple models’ strengths          | Requires careful tuning, computationally expensive |
| Voting Classifier  | Ensemble of different classifiers         | Improves accuracy, works with any classifier | Performance depends on model diversity |
| Extra Trees       | Large datasets, reducing overfitting       | Faster than Random Forest, robust to noise | Less interpretable than Random Forest  |
| Histogram-Based GB | Large datasets, high-speed training       | Fast training, efficient                 | May not be as flexible as XGBoost      |
'''
