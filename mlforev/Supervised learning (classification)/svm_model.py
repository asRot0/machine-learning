"""
Support Vector Machine (SVM) Model for Classification

Goal:
Train and evaluate a Support Vector Machine (SVM) model on preprocessed data using scikit-learn.

Objectives:
- Load and preprocess the dataset.
- Train the SVM model on the training data.
- Evaluate the model on the test data and output the classification report.
"""

from sklearn.svm import SVC
from sklearn.metrics import classification_report
from data_processing import DataProcessor


class SVMModel:
    """
    A class that encapsulates the Support Vector Machine (SVM) model for training and evaluation.
    """
    def __init__(self, kernel='rbf', C=1.0, gamma='scale'):
        """
        Initializes the SVM model with the specified kernel, regularization parameter C, and gamma.
        """
        self.model = SVC(kernel=kernel, C=C, gamma=gamma)

    def train(self, X_train, y_train):
        """
        Trains the SVM model using the provided training data.
        """
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        """
        Evaluates the SVM model on the test data.
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

    # Instantiate, train, and evaluate the SVM model
    svm_model = SVMModel(kernel='rbf', C=1.0, gamma='scale')
    svm_model.train(X_train, y_train)
    print(svm_model.evaluate(X_test, y_test))
