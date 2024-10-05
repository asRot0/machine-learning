from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np


class LinearRegressionModel:
    def __init__(self):
        self.model = LinearRegression()

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        score = self.model.score(X_test, y_test)
        y_pred = self.model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        return score, mse

    def plot_regression(self, X_train, y_train):
        plt.scatter(X_train, y_train, color="blue")
        x = np.linspace(min(X_train), max(X_train), 100)
        plt.plot(x, self.model.predict(x.reshape(-1, 1)), color="red", linewidth=3)
        plt.xlabel("Features")
        plt.ylabel("Target")
        plt.title("Linear Regression")
        plt.show()
