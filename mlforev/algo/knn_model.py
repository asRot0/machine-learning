"""
K-Nearest Neighbors (KNN) Model Script

Goal:
Train and evaluate a K-Nearest Neighbors model on preprocessed data.
"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from data_processing import DataProcessor


class KNNModel:
    def __init__(self, n_neighbors=5):
        """
        Initialize the KNN model with the specified number of neighbors.
        """
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors)

    def train(self, X_train, y_train):
        """
        Train the KNN model on the training data.
        """
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        """
        Evaluate the KNN model on test data and print the classification report.
        """
        y_pred = self.model.predict(X_test)
        return classification_report(y_test, y_pred)


# Example usage
if __name__ == "__main__":
    data_path = '../data/magic+gamma+telescope/magic04.data'
    columns = ["fLength", "fWidth", "fSize", "fConc", "fConc1", "fAsym", "fM3Long", "fM3Trans", "fAlpha", "fDist", "class"]

    processor = DataProcessor(data_path, columns, "class")
    train, valid, test = processor.split_data()

    # Scale and resample the datasets
    X_train, y_train = processor.scale_and_resample(train, oversample=True)
    X_valid, y_valid = processor.scale_and_resample(valid, oversample=False)
    X_test, y_test = processor.scale_and_resample(test, oversample=False)

    knn_model = KNNModel(n_neighbors=5)
    knn_model.train(X_train, y_train)
    print(knn_model.evaluate(X_test, y_test))
