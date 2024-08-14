import numpy as np
import pandas as pd

dataset = 'datasets/housing/housing.csv'
data = pd.read_csv(dataset)


def split_train_test(data, test_ratio):
    np.random.seed(42)
    shuffled_indices = np.random.permutation(len(data))
    print(shuffled_indices)
    test_set_size = int(len(data)*test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return train_indices, test_indices


train_set, test_set = split_train_test(data, 0.2)
print(len(data))
print(len(train_set), len(test_set))
