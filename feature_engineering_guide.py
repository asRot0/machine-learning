"""
Feature Engineering Guide for Machine Learning
==============================================
This script covers essential feature engineering techniques to extract,
transform, and create meaningful features for improving model performance.
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.preprocessing import PolynomialFeatures

# Example Dataset
data = {
    'Age': [25, 35, 45, 23, 54],
    'Salary': [50000, 60000, 80000, 45000, 100000],
    'Purchased': [0, 1, 0, 1, 1]  # 0 = No, 1 = Yes
}
df = pd.DataFrame(data)

print("Original Data:\n", df)

# 1. Feature Creation: Combining Existing Features
df['Age_Salary_Ratio'] = df['Age'] / df['Salary']
print("\nFeature Creation (Age to Salary Ratio):\n", df)

# 2. Binning or Discretization
df['Age_Bin'] = pd.cut(df['Age'], bins=[0, 30, 40, 60], labels=['Young', 'Middle-aged', 'Senior'])
print("\nBinning Age into Categories:\n", df)

# 3. Polynomial Features
poly = PolynomialFeatures(degree=2, include_bias=False)
numerical_features = df[['Age', 'Salary']]
poly_features = poly.fit_transform(numerical_features)
poly_feature_names = poly.get_feature_names_out(['Age', 'Salary'])

poly_df = pd.DataFrame(poly_features, columns=poly_feature_names)
print("\nPolynomial Features (Degree=2):\n", poly_df)

# 4. Interaction Features
df['Age_Salary_Product'] = df['Age'] * df['Salary']
print("\nFeature Interaction (Age * Salary):\n", df)

# 5. Feature Selection: SelectKBest with ANOVA F-statistic
X = df[['Age', 'Salary', 'Age_Salary_Ratio', 'Age_Salary_Product']]
y = df['Purchased']

selector = SelectKBest(score_func=f_classif, k=2)
X_new = selector.fit_transform(X, y)

selected_features = X.columns[selector.get_support()]
print("\nSelected Features (Using ANOVA F-statistic):\n", selected_features)

# 6. Feature Selection: Mutual Information
mi_scores = mutual_info_classif(X, y)
mi_df = pd.DataFrame({'Feature': X.columns, 'MI_Score': mi_scores})
print("\nMutual Information Scores:\n", mi_df)

# 7. Encoding Categorical Features Created (One-Hot Encoding)
encoded_df = pd.get_dummies(df, columns=['Age_Bin'], drop_first=True)
print("\nEncoded Categorical Features (One-Hot Encoding):\n", encoded_df)

# 8. Scaling and Normalizing New Features (Optional)
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(df[['Age_Salary_Ratio', 'Age_Salary_Product']])
df[['Age_Salary_Ratio_Scaled', 'Age_Salary_Product_Scaled']] = scaled_features
print("\nScaled New Features:\n", df)

# Final Dataset Ready for Model Training
print("\nFinal Feature-Engineered Dataset:\n", df)
