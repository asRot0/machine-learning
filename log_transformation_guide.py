import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew

# ===========================
# Sample Dataset
# ===========================
data = {
    "House_Price": [150000, 200000, 210000, 450000, 400000, 250000, 240000, 500000, 300000, 310000],
    "Income": [20, 25, 30, 40, 100, 200, 250, 500, 1000, 5000]  # Positively skewed data
}
df = pd.DataFrame(data)
print("Original Dataset:\n", df)

# ===========================
# Visualize the Original Distributions
# ===========================
plt.figure(figsize=(12, 5))

# Original Income Distribution
plt.subplot(1, 2, 1)
sns.histplot(df["Income"], kde=True, bins=10, color="blue")
plt.title("Original Income Distribution")
plt.xlabel("Income")

# Original House_Price Distribution
plt.subplot(1, 2, 2)
sns.histplot(df["House_Price"], kde=True, bins=10, color="orange")
plt.title("Original House_Price Distribution")
plt.xlabel("House_Price")

plt.tight_layout()
plt.show()

# ===========================
# Applying Log Transformation
# ===========================
# Add a small constant to avoid log(0)
df["Log_Income"] = np.log(df["Income"] + 1)
df["Log_House_Price"] = np.log(df["House_Price"] + 1)

# ===========================
# Visualizing Transformed Data
# ===========================
plt.figure(figsize=(12, 5))

# Log Transformed Income Distribution
plt.subplot(1, 2, 1)
sns.histplot(df["Log_Income"], kde=True, bins=10, color="green")
plt.title("Log Transformed Income Distribution")
plt.xlabel("Log(Income + 1)")

# Log Transformed House_Price Distribution
plt.subplot(1, 2, 2)
sns.histplot(df["Log_House_Price"], kde=True, bins=10, color="purple")
plt.title("Log Transformed House_Price Distribution")
plt.xlabel("Log(House_Price + 1)")

plt.tight_layout()
plt.show()

# ===========================
# Skewness Before and After
# ===========================
print("\nSkewness Before and After Log Transformation:")
for col in ["Income", "House_Price", "Log_Income", "Log_House_Price"]:
    col_skew = skew(df[col])
    print(f"{col} Skewness: {col_skew:.2f}")

"""
Key Points:
-----------
1. Original skewed distributions are normalized by log transformation.
2. Adding 1 ensures there are no issues with log(0).
3. Compare skewness before and after transformation to confirm normalization.
"""
