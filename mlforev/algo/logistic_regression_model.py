"""
Logistic Regression Model for Classification

Goal:
Train and evaluate a Logistic Regression model on preprocessed data using the scikit-learn library.

Objectives:
- Load and preprocess the dataset.
- Train the Logistic Regression model on the training data.
- Evaluate the model on the test data and output the classification report.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from data_processing import DataProcessor


class LogisticRegressionModel:
    """
    A class that encapsulates the Logistic Regression model for training and evaluation.
    """
    def __init__(self):
        """
        Initializes the Logistic Regression model.
        """
        self.model = LogisticRegression()

    def train(self, X_train, y_train):
        """
        Trains the Logistic Regression model using the provided training data.
        """
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        """
        Evaluates the Logistic Regression model on the test data.
        Outputs a classification report including precision, recall, and F1 score.
        """
        y_pred = self.model.predict(X_test)
        return classification_report(y_test, y_pred)


if __name__ == "__main__":
    # Define the data path and column names
    data_path = '../data/magic+gamma+telescope/magic04.data'
    columns = ["fLength", "fWidth", "fSize", "fConc", "fConc1", "fAsym", "fM3Long", "fM3Trans", "fAlpha", "fDist", "class"]

    # Instantiate the data processor
    processor = DataProcessor(data_path, columns, "class")

    # Split the dataset into training, validation, and test sets
    train, valid, test = processor.split_data()

    # Preprocess and scale the data, with oversampling if needed
    X_train, y_train = processor.scale_and_resample(train, oversample=True)
    X_valid, y_valid = processor.scale_and_resample(valid, oversample=False)
    X_test, y_test = processor.scale_and_resample(test, oversample=False)

    # Instantiate, train, and evaluate the Logistic Regression model
    lg_model = LogisticRegressionModel()
    lg_model.train(X_train, y_train)
    print(lg_model.evaluate(X_test, y_test))
