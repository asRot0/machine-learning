import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, StackingClassifier, VotingClassifier
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load dataset (example: Iris)
from sklearn.datasets import load_iris
data = load_iris()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 1️⃣ AdaBoost
adaboost = AdaBoostClassifier(n_estimators=50, random_state=42)

# 2️⃣ Gradient Boosting (GBM)
gbm = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)

# 3️⃣ XGBoost
xgb_model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, use_label_encoder=False, eval_metric='logloss', random_state=42)

# 4️⃣ LightGBM
lgbm = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.1, random_state=42)

# 5️⃣ CatBoost
catboost = CatBoostClassifier(iterations=100, learning_rate=0.1, depth=6, verbose=0, random_state=42)

# 6️⃣ HistGradientBoostingClassifier (Scikit-learn's histogram-based boosting)
hist_gbm = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.1, random_state=42)

# 7️⃣ Stacking Classifier
stacking = StackingClassifier(estimators=[
    ('xgb', xgb_model),
    ('lgbm', lgbm),
    ('catboost', catboost)
], final_estimator=LogisticRegression())

# 8️⃣ Voting Classifier
voting = VotingClassifier(estimators=[
    ('xgb', xgb_model), ('lgbm', lgbm), ('catboost', catboost)
], voting='hard')

# Train and Evaluate
models = {
    "AdaBoost": adaboost,
    "Gradient Boosting": gbm,
    "XGBoost": xgb_model,
    "LightGBM": lgbm,
    "CatBoost": catboost,
    "HistGradientBoost": hist_gbm,
    "Stacking Classifier": stacking,
    "Voting Classifier": voting
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{name}: Accuracy = {accuracy:.4f}")

''' 
| Algorithm            | Best For                                     | Pros                                          | Cons                          |
|---------------------|---------------------------------------------|----------------------------------------------|------------------------------|
| AdaBoost           | Small datasets, binary classification       | Simple, interpretable                        | Sensitive to noisy data       |
| Gradient Boosting  | Medium-sized datasets, regression           | Good accuracy, handles missing values       | Slower than LightGBM & XGBoost |
| XGBoost           | Large structured data, imbalanced datasets  | Fast, scalable, good accuracy               | Can overfit                   |
| LightGBM          | Very large datasets, real-time applications | Extremely fast, low memory                  | Not robust to outliers        |
| CatBoost          | Categorical data, finance, e-commerce       | Handles categorical features automatically  | Slower than LightGBM          |
| HistGradientBoost | Large datasets, fast training               | Optimized for speed and efficiency         | Less flexible than XGBoost    |
| LGBM Ranker       | Learning-to-rank tasks                      | Best for ranking problems like search engines | Needs large labeled data      |
| Stacking Classifier | Combining multiple models                   | Utilizes strengths of different models       | Computationally expensive      |
| Voting Classifier  | Blending multiple models for stability      | Simple and effective                         | Requires diverse classifiers  |
'''
