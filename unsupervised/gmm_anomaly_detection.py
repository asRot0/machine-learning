# gmm_anomaly_detection.py
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# Generate synthetic data
X, _ = make_blobs(n_samples=300, centers=1, cluster_std=0.5, random_state=42)

# Fit GMM and detect anomalies
gmm = GaussianMixture(n_components=1, random_state=42)
gmm.fit(X)
scores = gmm.score_samples(X)
anomalies = X[scores < -2.5]

# Plot data and anomalies
plt.scatter(X[:, 0], X[:, 1], c='blue', label='Normal Data')
plt.scatter(anomalies[:, 0], anomalies[:, 1], c='red', label='Anomalies')
plt.legend()
plt.title("Anomaly Detection with GMM")
plt.show()
