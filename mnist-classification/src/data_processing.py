import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

def load_data():
    mnist = fetch_openml('mnist_784', version=1)
    X, y = mnist['data'], mnist['target'].astype(np.uint8)
    return X, y

def split_data(X, y, test_size=0.2):
    return train_test_split(X, y, test_size=test_size, random_state=42)
