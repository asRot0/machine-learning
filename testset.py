from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

os.environ['COLUMNS'] = '200'
dataset = 'datasets/housing/housing.csv'

housing = pd.read_csv(dataset)

# train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

# housing['median_income'].hist()
# plt.show()

housing['income_cat'] = np.ceil(housing['median_income'] / 1.5)
print(housing.head())
housing['income_cat'].where(housing['income_cat'] < 5, 5.0, inplace=True)
print(housing.head())

# housing['income_cat'].hist()
# plt.show()

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
print(split)

strat_train_set = None
strat_test_set = None

for train_index, test_index in split.split(housing, housing['income_cat']):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

print(housing['income_cat'].value_counts() / len(housing))

for set_ in (strat_train_set, strat_test_set):
    set_.drop('income_cat', axis=1, inplace=True)
# print(strat_test_set.head())

housing = strat_train_set.copy()
housing.plot(kind='scatter', x='longitude', y='latitude', alpha=0.1)
plt.show()