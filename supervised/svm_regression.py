import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import LinearSVR, SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

# Load the dataset
data = fetch_california_housing()
X = data.data
y = data.target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()

# Linear SVM Regression (LinearSVR)
linear_svr_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svr', LinearSVR(epsilon=0.1, random_state=42, max_iter=10000))
])

# Fit and evaluate LinearSVR
linear_svr_pipeline.fit(X_train, y_train)
y_pred_linear = linear_svr_pipeline.predict(X_test)
mse_linear = mean_squared_error(y_test, y_pred_linear)
print("LinearSVR MSE:", mse_linear)

# SVR with different kernels and hyperparameter tuning
param_grid = {
    'svr__C': [0.1, 1, 10],
    'svr__epsilon': [0.01, 0.1, 0.5],
    'svr__kernel': ['poly', 'rbf'],
    'svr__degree': [2, 3],  # Only used if kernel='poly'
}

svr_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svr', SVR())
])

grid_search = GridSearchCV(svr_pipeline, param_grid, cv=3, scoring='neg_mean_squared_error', verbose=1)
grid_search.fit(X_train, y_train)

# Evaluate the best model
best_svr = grid_search.best_estimator_
y_pred_svr = best_svr.predict(X_test)
mse_svr = mean_squared_error(y_test, y_pred_svr)
print("Best SVR MSE:", mse_svr)
print("Best Parameters:", grid_search.best_params_)
