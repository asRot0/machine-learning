"""
Dimensionality Reduction with Principal Component Analysis (PCA)
=================================================================
This script demonstrates how to apply PCA to reduce the dimensionality of a dataset
and visualize the results in 2D or 3D space.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification

# ===========================================
# 1. Load and Prepare the Data
# ===========================================

# Using the Iris dataset as an example
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
class_names = iris.target_names

# Standardize the data (PCA works better with scaled data)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Original Dataset Shape:", X_scaled.shape)

# ===========================================
# 2. Apply PCA for Dimensionality Reduction
# ===========================================

# Reduce the dataset to 2 dimensions
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Explained variance ratio for each principal component
explained_variance_ratio = pca.explained_variance_ratio_
print("\nExplained Variance Ratio (2 Components):", explained_variance_ratio)
print("Total Explained Variance (2 Components):", np.sum(explained_variance_ratio))

# ===========================================
# 3. Visualize the Data in 2D Space
# ===========================================

# Scatter plot of PCA-transformed data
plt.figure(figsize=(8, 6))
for class_index, class_name in enumerate(class_names):
    plt.scatter(X_pca[y == class_index, 0], X_pca[y == class_index, 1], label=class_name)
plt.title("PCA on Iris Dataset (2D)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.grid(True)
plt.show()

# ===========================================
# 4. Apply PCA for 3D Visualization (Optional)
# ===========================================

# Reduce the dataset to 3 dimensions
pca_3d = PCA(n_components=3)
X_pca_3d = pca_3d.fit_transform(X_scaled)

# Plot in 3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
for class_index, class_name in enumerate(class_names):
    ax.scatter(X_pca_3d[y == class_index, 0],
               X_pca_3d[y == class_index, 1],
               X_pca_3d[y == class_index, 2], label=class_name)
ax.set_title("PCA on Iris Dataset (3D)")
ax.set_xlabel("Principal Component 1")
ax.set_ylabel("Principal Component 2")
ax.set_zlabel("Principal Component 3")
ax.legend()
plt.show()

# ===========================================
# 5. Determine Optimal Number of Components
# ===========================================

# Calculate explained variance for all components
pca_full = PCA()
X_pca_full = pca_full.fit_transform(X_scaled)
cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)

# Plot cumulative explained variance
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='--', color='b')
plt.title("Explained Variance by Number of Principal Components")
plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Explained Variance")
plt.grid(True)
plt.axhline(y=0.95, color='r', linestyle='--', label="95% Variance Threshold")
plt.legend()
plt.show()

# ===========================================
# 6. Real-World Use Case: Synthetic Dataset
# ===========================================

# Generate a synthetic dataset with 10 features
X_synthetic, y_synthetic = make_classification(n_samples=500, n_features=10, n_classes=3, n_informative=5, random_state=42)

# Standardize the synthetic dataset
X_synthetic_scaled = scaler.fit_transform(X_synthetic)

# Apply PCA to reduce to 2 components
pca_synthetic = PCA(n_components=2)
X_synthetic_pca = pca_synthetic.fit_transform(X_synthetic_scaled)

# Visualize synthetic data in 2D
plt.figure(figsize=(8, 6))
for class_index in np.unique(y_synthetic):
    plt.scatter(X_synthetic_pca[y_synthetic == class_index, 0],
                X_synthetic_pca[y_synthetic == class_index, 1], label=f"Class {class_index}")
plt.title("PCA on Synthetic Dataset (2D)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.grid(True)
plt.show()

# ===========================================
# Summary
# ===========================================
"""
Key Concepts Covered:
1. PCA for dimensionality reduction.
2. Visualizing data in reduced dimensions (2D and 3D).
3. Determining the optimal number of components using explained variance.
4. Application of PCA on real-world and synthetic datasets.
"""
