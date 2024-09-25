"""
Data Preprocessing Script

Goal:
This script handles loading the dataset, cleaning, scaling, splitting, and oversampling if necessary.
It returns preprocessed data ready for training and testing machine learning models.
"""

from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class DataProcessor:
    def __init__(self, data_path, columns, target_col):
        """
        Initialize the data processor with dataset path, columns, and target column (class label).
        """
        self.data = pd.read_csv(data_path, names=columns)
        self.target_col = target_col
        self.data[target_col] = (self.data[target_col] == 'g').astype(int)  # Convert target to binary

    def plot_data_distribution(self):
        """
        Plots the distribution of features for each class.
        """
        for label in self.data.columns[:-1]:
            plt.hist(self.data[self.data[self.target_col] == 1][label], color='lime', label='gamma', alpha=0.7, density=True)
            plt.hist(self.data[self.data[self.target_col] == 0][label], color='orange', label='hadron', alpha=0.7, density=True)
            plt.title(label)
            plt.xlabel(label)
            plt.legend()
            plt.show()

    def split_data(self):
        """
        Split the dataset into training, validation, and test sets.
        :return: train, valid, test dataframes
        """
        return np.split(self.data.sample(frac=1), [int(0.6 * len(self.data)), int(0.8 * len(self.data))])

    def scale_and_resample(self, dataframe, oversample=False):
        """
        Scale the features and optionally oversample the dataset.
        :param dataframe: Input dataframe to scale and optionally resample.
        :param oversample: Boolean flag for oversampling.
        :return: scaled and possibly oversampled dataset
        """
        X = dataframe[dataframe.columns[:-1]].values
        y = dataframe[dataframe.columns[-1]].values

        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        if oversample:
            ros = RandomOverSampler()
            X, y = ros.fit_resample(X, y)

        return X, y
