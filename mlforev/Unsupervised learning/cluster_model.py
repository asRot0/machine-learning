from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


class ClusteringModel:
    def __init__(self, n_clusters):
        """
        Initialize the clustering model with KMeans.
        :param n_clusters: Number of clusters
        """
        self.kmeans = KMeans(n_clusters=n_clusters)

    def fit(self, X):
        """
        Fit the KMeans model on the data.
        :param X: Features for clustering
        """
        self.kmeans.fit(X)

    def predict(self, X):
        """
        Predict cluster labels for the given data.
        :param X: Features for prediction
        :return: Predicted cluster labels
        """
        return self.kmeans.predict(X)

    def visualize_clusters(self, X, x_label, y_label):
        """
        Visualize the clusters with a scatter plot.
        :param X: Features
        :param x_label: X-axis label
        :param y_label: Y-axis label
        """
        clusters = self.kmeans.labels_
        cluster_df = pd.DataFrame(np.hstack((X, clusters.reshape(-1, 1))), columns=[x_label, y_label, "class"])
        sns.scatterplot(x=x_label, y=y_label, hue='class', data=cluster_df)
        plt.show()
