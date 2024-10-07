import pandas as pd


class DataProcessor:
    def __init__(self, file_path, cols):
        """
        Initialize the data processor with the dataset.
        :param file_path: Path to the dataset
        :param cols: List of column names
        """
        self.cols = cols
        self.df = pd.read_csv(file_path, names=cols, sep="\s+")

    def preview_data(self):
        """
        Preview the first 5 rows of the dataset.
        """
        return self.df.head()

    def get_features_and_labels(self, x_cols, y_col):
        """
        Split the dataset into features (X) and labels (y).
        :param x_cols: List of column names for features
        :param y_col: Column name for the target label
        """
        X = self.df[x_cols].values
        y = self.df[y_col].values
        return X, y
