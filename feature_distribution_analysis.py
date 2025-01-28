"""
Feature Distribution Analysis for Machine Learning
===================================================
This script demonstrates how to analyze and visualize feature distributions
to understand whether the data is symmetric or skewed. It includes:
1. Histogram and KDE plots
2. Box plots for outliers
3. Skewness calculation for numerical features
4. Recommendations for imputation strategies based on distribution

"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import skew

# ===========================
# 1. Sample Dataset Creation
# ===========================

data = {
    "Age": [25, 32, 40, 47, 52, 29, 31, 45, 38, np.nan],
    "Salary": [50000, 60000, 52000, 75000, 80000, 62000, 62000, 70000, np.nan, 58000],
    "House_Price": [150000, 200000, 210000, 450000, 400000, 250000, 240000, 500000, 300000, 310000],
}

df = pd.DataFrame(data)
print("Original Dataset:\n", df)


# ============================
# 2. Visualizing Distributions
# ============================

def plot_distribution_and_skew(df):
    """
    Function to plot distributions (Histogram + KDE) and Box Plots
    for all numerical columns in a DataFrame. Calculates skewness.

    Parameters:
        df (pd.DataFrame): The DataFrame containing numerical features.
    """
    for column in df.select_dtypes(include=[np.number]).columns:
        plt.figure(figsize=(12, 6))

        # Histogram + KDE Plot
        sns.histplot(df[column], kde=True, bins=10, color='blue', stat='density')
        plt.title(f"Distribution of {column}", fontsize=16)
        plt.xlabel(column, fontsize=12)
        plt.ylabel("Density", fontsize=12)
        plt.grid(axis='y', alpha=0.75)

        # Box Plot for Outliers
        plt.figure(figsize=(6, 4))
        sns.boxplot(x=df[column], color='orange')
        plt.title(f"Box Plot of {column}", fontsize=16)
        plt.xlabel(column, fontsize=12)
        plt.show()

        # Skewness Calculation
        column_skew = skew(df[column].dropna())  # Drop NaN for skewness calculation
        print(f"{column} Skewness: {column_skew:.2f}")
        if column_skew > 0.5:
            print(f"{column} is positively skewed.\n")
        elif column_skew < -0.5:
            print(f"{column} is negatively skewed.\n")
        else:
            print(f"{column} is approximately symmetric.\n")


# Call the function
plot_distribution_and_skew(df)

# ============================
# 3. Handling Missing Values
# ============================

print("\n--- Handling Missing Values ---")

# Strategy based on distribution
for column in df.columns:
    if column == "Age":
        # Age is approximately symmetric (use mean imputation)
        df[column].fillna(df[column].mean(), inplace=True)
        print(f"{column}: Mean imputation applied.")
    elif column == "Salary":
        # Salary is slightly skewed (use median imputation)
        df[column].fillna(df[column].median(), inplace=True)
        print(f"{column}: Median imputation applied.")

print("\nAfter Imputation:\n", df)

# ============================
# 4. Recommendations Summary
# ============================

print("\n--- Recommendations Summary ---")
for column in df.columns:
    column_skew = skew(df[column]) if column in df.select_dtypes(include=[np.number]).columns else None
    if column_skew is not None:
        if column_skew > 0.5:
            print(f"{column}: Positively skewed. Consider using median imputation or transformations (e.g., log).")
        elif column_skew < -0.5:
            print(f"{column}: Negatively skewed. Consider using median imputation.")
        else:
            print(f"{column}: Symmetric. Mean imputation is suitable.")
    else:
        print(f"{column}: Non-numerical or categorical data.")

# ============================
# 5. Final Dataset
# ============================

print("\nFinal Dataset After Analysis and Imputation:\n", df)

"""
Output Insights:
----------------
1. Distributions:
   - Histograms and KDE plots provide visual insights into the shape of the data.
   - Box plots help identify outliers.
2. Skewness:
   - Symmetric (Skewness ≈ 0): Use mean for imputation.
   - Positively skewed (Skewness > 0.5): Use median for imputation or log transformation.
   - Negatively skewed (Skewness < -0.5): Use median for imputation.
3. Handling Missing Values:
   - Imputation strategy depends on distribution shape.
4. Recommendations:
   - Adjust feature scaling or transformations for better model performance.

"""
