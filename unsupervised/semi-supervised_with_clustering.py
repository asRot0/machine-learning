# semi_supervised_with_clustering.py
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# Load the digits dataset
X_digits, y_digits = load_digits(return_X_y=True)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_digits, y_digits, test_size=0.3, random_state=42)

# -------------------
# Step 1: Baseline Logistic Regression Model
# -------------------
# Train logistic regression on the entire labeled training data
log_reg = LogisticRegression(random_state=42, max_iter=2000)
log_reg.fit(X_train, y_train)
print('Baseline Logistic Regression Score:', log_reg.score(X_test, y_test))

# -------------------
# Step 2: Logistic Regression with KMeans Clustering
# -------------------
# Pipeline with KMeans clustering followed by logistic regression
pipeline = Pipeline([
    ('kmeans', KMeans(n_clusters=50, random_state=42)),
    ('log_reg', LogisticRegression(random_state=42, max_iter=2000))
])
pipeline.fit(X_train, y_train)
print('Score After Clustering:', pipeline.score(X_test, y_test))

# -------------------
# Step 3: Semi-Supervised Learning with Few Labeled Samples
# -------------------
# Train logistic regression with only a few labeled examples
n_labeled = 50  # Number of labeled samples
log_reg_few = LogisticRegression(max_iter=2000, random_state=42)
log_reg_few.fit(X_train[:n_labeled], y_train[:n_labeled])
print('Score with Few Labeled Samples:', log_reg_few.score(X_test, y_test))

# -------------------
# Step 4: Representative Samples from KMeans Clusters
# -------------------
# Train KMeans and find representative points from each cluster
k = 50  # Number of clusters
kmeans = KMeans(n_clusters=k, random_state=42)
X_digits_dist = kmeans.fit_transform(X_train)
representative_digit_idx = np.argmin(X_digits_dist, axis=0)
X_representative_digits = X_train[representative_digit_idx]

# Assign labels manually for representative samples
y_representative_digits = np.array([4, 8, 0, 6, 8, 3, 7, 7, 9, 2,
                                    5, 5, 8, 5, 2, 1, 2, 9, 6, 1,
                                    1, 6, 9, 0, 8, 3, 0, 7, 4, 1,
                                    6, 5, 2, 4, 1, 8, 6, 3, 9, 2,
                                    4, 2, 9, 4, 7, 6, 2, 3, 1, 1])

# Train logistic regression on these representative points
log_reg_rep = LogisticRegression(max_iter=2000, random_state=42)
log_reg_rep.fit(X_representative_digits, y_representative_digits)
print('Score After Training on KMeans Representative Samples:', log_reg_rep.score(X_test, y_test))

# -------------------
# Step 5: Propagate Labels within Each Cluster
# -------------------
# Label propagation within each cluster using the representative labels
y_train_propagated = np.empty(len(X_train), dtype=np.int32)
for i in range(k):
    y_train_propagated[kmeans.labels_ == i] = y_representative_digits[i]

# Train logistic regression on the fully labeled training set after propagation
log_reg_prop = LogisticRegression(max_iter=2000, random_state=42)
log_reg_prop.fit(X_train, y_train_propagated)
print('Score After Label Propagation:', log_reg_prop.score(X_test, y_test))

# -------------------
# Step 6: Partial Propagation with Cutoff Distance
# -------------------
# Propagate labels only for points within a certain percentile distance from each cluster center
percentile_closest = 20  # Use top 20% closest points in each cluster
X_cluster_dist = X_digits_dist[np.arange(len(X_train)), kmeans.labels_]
for i in range(k):
    in_cluster = (kmeans.labels_ == i)
    cluster_dist = X_cluster_dist[in_cluster]
    cutoff_distance = np.percentile(cluster_dist, percentile_closest)
    X_cluster_dist[in_cluster & (X_cluster_dist > cutoff_distance)] = -1  # Exclude points above cutoff

# Select partially propagated points
partially_propagated = (X_cluster_dist != -1)
X_train_partially_propagated = X_train[partially_propagated]
y_train_partially_propagated = y_train_propagated[partially_propagated]

# Train logistic regression on partially propagated labels
log_reg_partial = LogisticRegression(max_iter=2000, random_state=42)
log_reg_partial.fit(X_train_partially_propagated, y_train_partially_propagated)
print('Score After Partial Propagation:', log_reg_partial.score(X_test, y_test))

# Evaluate the consistency of partially propagated labels with true labels
consistency = np.mean(y_train_partially_propagated == y_train[partially_propagated])
print('Consistency of Partially Propagated Labels with True Labels:', consistency)
