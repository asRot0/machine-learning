from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

class PCAProcessor:
    def __init__(self, n_components):
        """
        Initialize the PCA processor.
        :param n_components: Number of components for PCA
        """
        self.pca = PCA(n_components=n_components)

    def fit_transform(self, X):
        """
        Fit and transform the data using PCA.
        :param X: Features
        :return: Transformed PCA data
        """
        return self.pca.fit_transform(X)

    def visualize_pca(self, X_pca, labels):
        """
        Visualize PCA components.
        :param X_pca: PCA-transformed data
        :param labels: Original or predicted class labels
        """
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels)
        plt.xlabel("PCA1")
        plt.ylabel("PCA2")
        plt.show()
