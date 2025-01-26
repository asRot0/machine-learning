"""
Data Preprocessing Guide for Machine Learning
==============================================
This script covers the essential steps to preprocess data before feeding it into a machine learning model.
Includes techniques like handling missing data, scaling, encoding, and splitting datasets.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Example Dataset
data = {
    'Age': [25, np.nan, 35, 45, 29, np.nan],
    'Salary': [50000, 60000, 80000, np.nan, 40000, 30000],
    'Country': ['USA', 'France', 'USA', 'Germany', 'Germany', 'France'],
    'Purchased': ['No', 'Yes', 'No', 'Yes', 'No', 'Yes']
}

df = pd.DataFrame(data)

# 1. Handling Missing Data
print("Original Data:\n", df)

imputer = SimpleImputer(strategy='mean')
df['Age'] = imputer.fit_transform(df[['Age']])
df['Salary'] = imputer.transform(df[['Salary']])
print("\nData After Handling Missing Values:\n", df)

# 2. Encoding Categorical Data
encoder = OneHotEncoder()
country_encoded = encoder.fit_transform(df[['Country']]).toarray()
country_labels = encoder.categories_[0]
encoded_df = pd.DataFrame(country_encoded, columns=country_labels)
df = df.drop('Country', axis=1).join(encoded_df)
print("\nData After Encoding Categorical Values:\n", df)

# 3. Feature Scaling
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df[['Age', 'Salary']])
df[['Age', 'Salary']] = scaled_features
print("\nData After Feature Scaling:\n", df)

# 4. Splitting Data into Train and Test Sets
X = df.drop('Purchased', axis=1)
y = df['Purchased']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("\nTrain-Test Split:")
print("X_train:\n", X_train)
print("X_test:\n", X_test)
print("y_train:\n", y_train)
print("y_test:\n", y_test)

# Optional: Use a Pipeline for Streamlined Preprocessing
categorical_features = ['Country']
numerical_features = ['Age', 'Salary']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='mean'), numerical_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('scaler', StandardScaler())
])

# Apply pipeline on example data
processed_data = pipeline.fit_transform(df)
print("\nPipeline Processed Data:\n", processed_data)
