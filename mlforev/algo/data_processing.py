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


class DataProcessor:
    def __init__(self, data_path, columns, target_col):
        """
        Initialize the data processor with dataset path, columns, and target column (class label).
        """
        self.data = pd.read_csv(data_path, names=columns)
        self.target_col = target_col
        self.data[target_col] = (self.data[target_col] == 'g').astype(int)  # Convert target to binary

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
