"""
Clustering with k-Means
========================
This script demonstrates the k-Means clustering algorithm using a sample dataset.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score

# ===========================================
# 1. Generate Synthetic Data
# ===========================================

# Create synthetic data with 4 clusters
n_samples = 500
n_features = 2
n_clusters = 4
random_state = 42

X, y_true = make_blobs(n_samples=n_samples, centers=n_clusters,
                       n_features=n_features, random_state=random_state)

# Visualize the generated data
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c='gray', s=30, label='Data Points')
plt.title("Generated Data")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()

# ===========================================
# 2. Apply k-Means Clustering
# ===========================================

# Initialize k-Means
kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)

# Fit k-Means and predict clusters
y_kmeans = kmeans.fit_predict(X)

# Visualize the clusters
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=30, cmap='viridis', label='Clustered Data')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            c='red', s=200, marker='X', label='Centroids')
plt.title("k-Means Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()

# ===========================================
# 3. Evaluate the Clustering
# ===========================================

# Inertia (Sum of squared distances to nearest centroid)
print(f"Inertia: {kmeans.inertia_:.2f}")

# Silhouette Score
sil_score = silhouette_score(X, y_kmeans)
print(f"Silhouette Score: {sil_score:.2f}")

# ===========================================
# 4. Choosing Optimal k Using the Elbow Method
# ===========================================

inertia_values = []
k_values = range(1, 10)

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=random_state)
    kmeans.fit(X)
    inertia_values.append(kmeans.inertia_)

# Plot the elbow curve
plt.figure(figsize=(8, 6))
plt.plot(k_values, inertia_values, marker='o', linestyle='--', color='b')
plt.title("Elbow Method to Determine Optimal k")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.xticks(k_values)
plt.grid(True)
plt.show()

# ===========================================
# 5. Real-World Use Case: Customer Segmentation
# ===========================================

# Simulating customer data (e.g., annual income and spending score)
np.random.seed(random_state)
n_customers = 300
income = np.random.randint(20, 150, n_customers)
spending_score = np.random.randint(1, 100, n_customers)

# Create a DataFrame
customer_data = pd.DataFrame({
    'Annual_Income': income,
    'Spending_Score': spending_score
})

# Apply k-Means for customer segmentation
kmeans = KMeans(n_clusters=5, random_state=random_state)
customer_data['Cluster'] = kmeans.fit_predict(customer_data)

# Visualize customer clusters
plt.figure(figsize=(8, 6))
for cluster in range(5):
    cluster_data = customer_data[customer_data['Cluster'] == cluster]
    plt.scatter(cluster_data['Annual_Income'], cluster_data['Spending_Score'],
                label=f"Cluster {cluster}")
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            c='red', s=200, marker='X', label='Centroids')
plt.title("Customer Segmentation")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score")
plt.legend()
plt.grid(True)
plt.show()

# ===========================================
# Summary
# ===========================================
"""
Key Concepts Covered:
1. Generated synthetic data for clustering.
2. Applied k-Means to find clusters and visualized them.
3. Evaluated clustering using inertia and silhouette score.
4. Determined optimal k using the elbow method.
5. Demonstrated a real-world use case for customer segmentation.
"""
