from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# os.environ['COLUMNS'] = '200'
dataset = 'datasets/housing/housing.csv'

housing = pd.read_csv(dataset)

# train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

# housing['median_income'].hist()
# plt.show()
print(housing.info())

housing['income_cat'] = np.ceil(housing['median_income'] / 1.5)
print(housing.head())
housing['income_cat'].where(housing['income_cat'] < 5, 5.0, inplace=True)
print(housing.head())

# housing['income_cat'].hist()
# plt.show()

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
print(split)

strat_train_set = pd.DataFrame
strat_test_set = pd.DataFrame

for train_index, test_index in split.split(housing, housing['income_cat']):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

print(housing['income_cat'].value_counts() / len(housing))

for set_ in (strat_train_set, strat_test_set):
    set_.drop('income_cat', axis=1, inplace=True)
# print(strat_test_set.head())

housing = pd.DataFrame(strat_train_set.copy())
'''
housing.plot(kind='scatter', x='longitude', y='latitude', alpha=0.4,
             s=housing['population']/100, label='population', figsize=(10,7),
             c='median_house_value', cmap=plt.get_cmap('jet'), colorbar=True)
plt.legend()
plt.show()
'''

housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"] = housing["population"]/housing["households"]


# housing.drop('ocean_proximity', axis=1, inplace=True)
corr_matrix = housing.corr(numeric_only=True)
# corr_matrix = housing.select_dtypes(include=[float, int]).corr()
print('correlations -----------------')
print(corr_matrix['median_house_value'].sort_values(ascending=False))

housing = strat_train_set.drop('median_house_value', axis=1)
housing_labels = strat_train_set['median_house_value'].copy()

'''
median = housing['total_bedrooms'].median()
housing['total_bedrooms'].fillna(median, inplace=True)
print(median, type(median))
'''
# print(housing.drop('ocean_proximity', axis=1).median().values)

imputer = SimpleImputer(strategy='median')
housing_num = housing.drop('ocean_proximity', axis=1)

imputer.fit(housing_num)
print(imputer.statistics_)

X = imputer.transform(housing_num)
housing_tr = pd.DataFrame(X, columns=housing_num.columns)

housing_cat = housing[['ocean_proximity']]
print(housing_cat.head())