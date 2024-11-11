# anomaly_detection.py
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# Generate synthetic data
X, _ = make_blobs(n_samples=300, centers=1, cluster_std=0.5, random_state=42)

# Apply Isolation Forest
iso_forest = IsolationForest(contamination=0.1, random_state=42)
y_pred_iso = iso_forest.fit_predict(X)
anomalies_iso = X[y_pred_iso == -1]

# Apply LOF
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
y_pred_lof = lof.fit_predict(X)
anomalies_lof = X[y_pred_lof == -1]

# Plot results
plt.scatter(X[:, 0], X[:, 1], color='blue', label='Normal Data')
plt.scatter(anomalies_iso[:, 0], anomalies_iso[:, 1], color='red', label='Anomalies (IF)')
plt.scatter(anomalies_lof[:, 0], anomalies_lof[:, 1], color='green', marker='x', label='Anomalies (LOF)')
plt.legend()
plt.title("Anomaly Detection with Isolation Forest and LOF")
plt.savefig('../plotfig/anomaly_detection.png')
plt.show()
