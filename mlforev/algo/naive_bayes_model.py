"""
Naive Bayes Model Script

Goal:
Train and evaluate a Naive Bayes classifier on preprocessed data.
"""

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from data_processing import DataProcessor


class NaiveBayesModel:
    def __init__(self):
        """
        Initialize the Naive Bayes model.
        """
        self.model = GaussianNB()

    def train(self, X_train, y_train):
        """
        Train the Naive Bayes model on the training data.
        """
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        """
        Evaluate the Naive Bayes model on test data and print the classification report.
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

    nb_model = NaiveBayesModel()
    nb_model.train(X_train, y_train)
    print(nb_model.evaluate(X_test, y_test))
