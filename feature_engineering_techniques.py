"""
Feature Engineering Techniques for Machine Learning
====================================================
This script demonstrates various feature engineering techniques,
including handling missing values, encoding categorical variables,
scaling numerical features, and generating new features.
"""

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

# Sample Dataset Creation
data = {
    "Age": [25, 32, np.nan, 47, 52, np.nan, 31, 45],
    "Salary": [50000, 60000, 52000, 75000, 80000, np.nan, 62000, 70000],
    "Gender": ["Male", "Female", "Female", "Male", "Male", "Female", "Female", "Male"],
    "Purchased": ["No", "Yes", "No", "Yes", "Yes", "No", "Yes", "No"]
}
df = pd.DataFrame(data)
print("Original Dataset:\n", df)

# Splitting features and target variable
X = df.drop(columns=["Purchased"])
y = df["Purchased"]

# ==========================
# 1. Handling Missing Values
# ==========================

# Strategy 1: Simple Imputation (mean, median, most_frequent)

# Select numerical columns (Age and Salary)
numerical_columns = ["Age", "Salary"]

simple_imputer = SimpleImputer(strategy="mean")
X[numerical_columns] = simple_imputer.fit_transform(X[numerical_columns])

print("\nAfter Simple Imputation:\n", X)

# Strategy 2: K-Nearest Neighbors Imputation
knn_imputer = KNNImputer(n_neighbors=2)
X_knn_imputed = knn_imputer.fit_transform(X.select_dtypes(include=["float64"]))
X_knn_df = pd.DataFrame(X_knn_imputed, columns=["Age", "Salary"])
print("\nKNN Imputation:\n", X_knn_df)

# ============================
# 2. Encoding Categorical Data
# ============================

# One-Hot Encoding
ohe = OneHotEncoder(sparse_output=False)
gender_encoded = ohe.fit_transform(X[["Gender"]])
gender_encoded_df = pd.DataFrame(gender_encoded, columns=ohe.get_feature_names_out(["Gender"]))

X_encoded = pd.concat([X.drop(columns=["Gender"]), gender_encoded_df], axis=1)
print("\nAfter One-Hot Encoding:\n", X_encoded)

# Label Encoding
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
print("\nEncoded Target Variable:\n", y_encoded)

# =======================
# 3. Feature Scaling
# =======================

# Min-Max Scaling
min_max_scaler = MinMaxScaler()
X_scaled_minmax = min_max_scaler.fit_transform(X_encoded)
print("\nMin-Max Scaled Features:\n", pd.DataFrame(X_scaled_minmax, columns=X_encoded.columns))

# Standard Scaling (z-score normalization)
standard_scaler = StandardScaler()
X_scaled_standard = standard_scaler.fit_transform(X_encoded)
print("\nStandard Scaled Features:\n", pd.DataFrame(X_scaled_standard, columns=X_encoded.columns))

# ===============================
# 4. Feature Selection Techniques
# ===============================

# Univariate Feature Selection
selector = SelectKBest(score_func=f_classif, k=2)
X_selected = selector.fit_transform(X_encoded, y_encoded)
print("\nSelected Features (Top 2 by ANOVA F-Test):\n", X_selected)

# Feature Importance (Tree-Based Models)
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(random_state=42)
model.fit(X_encoded, y_encoded)

feature_importances = pd.DataFrame({
    "Feature": X_encoded.columns,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)
print("\nFeature Importances:\n", feature_importances)

# ===============================
# 5. Dimensionality Reduction
# ===============================

# Principal Component Analysis (PCA)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled_standard)
print("\nPCA-Reduced Features:\n", X_pca)

# Plot PCA Variance Explained
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
plt.bar(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_)
plt.xlabel("Principal Components")
plt.ylabel("Variance Explained")
plt.title("Explained Variance Ratio by PCA Components")
plt.show()

# ==================================
# 6. Generating New Features
# ==================================

# Example 1: Polynomial Features
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X_scaled_standard)
print("\nPolynomial Features (Degree 2):\n", X_poly)

# Example 2: Binning
X["Age_Bin"] = pd.cut(X["Age"], bins=[0, 30, 50, 80], labels=["Young", "Middle-Aged", "Senior"])
print("\nAfter Binning Age:\n", X)

# Example 3: Interaction Features
X["Salary_Age_Interaction"] = X["Salary"] * X["Age"]
print("\nAfter Interaction Feature:\n", X)

# Final Split and Model Training
X_final = pd.concat([X_encoded, X[["Age_Bin", "Salary_Age_Interaction"]]], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X_final, y_encoded, test_size=0.2, random_state=42)
print("\nFinal Dataset for Training and Testing:\n", X_final.head())


'''
Handling Missing Values:
    - Simple Imputation (Mean, Median).
    - KNN-Based Imputation.
Encoding Categorical Variables:
    - One-Hot Encoding.
    - Label Encoding.
Feature Scaling:
    - Min-Max Scaling.
    - Standard Scaling (Z-Score Normalization).
Feature Selection:
    - ANOVA F-Test with SelectKBest.
    - Feature Importance using Tree-Based Models.
Dimensionality Reduction:
    - PCA to reduce dimensions while retaining variance.
    - Visualizing explained variance.
Generating New Features:
    - Polynomial Features for non-linear relations.
    - Binning for categorical grouping.
    - Interaction terms for combined effects.
'''