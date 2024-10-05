import pandas as pd
import numpy as np


class DataProcessor:
    def __init__(self, data_path, y_label, drop_cols=None, sample_hour=None):
        self.df = pd.read_csv(data_path).drop(["Date", "Holiday", "Seasons"], axis=1)
        self.y_label = y_label
        self.dataset_cols = ["bike_count", "hour", "temp", "humidity", "wind", "visibility", "dew_pt_temp", "radiation",
                             "rain", "snow", "functional"]
        self.df.columns = self.dataset_cols
        self.df["functional"] = (self.df["functional"] == 'Yes').astype(int)

        if drop_cols:
            self.df = self.df.drop(drop_cols, axis=1)
        if sample_hour:
            self.df = self.df[self.df['hour'] == sample_hour]
            self.df = self.df.drop(['hour'], axis=1)

    def data_info(self):
        """
        Display dataset summary, head, and missing values.
        """
        print(self.df.info())
        print(self.df.describe())
        print(self.df.head())

    def split_data(self, train_frac=0.6, val_frac=0.2):
        """
        Split the data into train, validation, and test sets.
        """
        train, val, test = np.split(self.df.sample(frac=1),
                                    [int(train_frac * len(self.df)), int((train_frac + val_frac) * len(self.df))])
        return train, val, test

    def get_xy(self, dataframe, x_labels=None):
        """
        Get X (features) and y (target) from the dataframe.
        """
        if x_labels is None:
            X = dataframe.drop([self.y_label], axis=1).values
        else:
            X = dataframe[x_labels].values
        y = dataframe[self.y_label].values.reshape(-1, 1)
        return X, y
